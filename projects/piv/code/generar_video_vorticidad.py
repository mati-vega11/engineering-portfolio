import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2

os.environ.setdefault("MPLCONFIGDIR", str(Path("resultados") / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import numpy as np

COOLWARM = matplotlib.colormaps["coolwarm"]


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_vectors(path: Path):
    current_key = None
    current_rows = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (int(row["frame_a"]), int(row["frame_b"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                yield {"frame_a": current_key[0], "frame_b": current_key[1], "rows": current_rows}
                current_key = key
                current_rows = []
            if row.get("valid", "0") == "1":
                current_rows.append(
                    {
                        "x_px": float(row["x_px"]),
                        "y_px": float(row["y_px"]),
                        "x_mm": float(row["x_mm"]),
                        "y_mm": float(row["y_mm"]),
                        "u_mm_s": float(row["u_mm_s"]),
                        "v_mm_s": float(row["v_mm_s"]),
                    }
                )
        if current_key is not None:
            yield {"frame_a": current_key[0], "frame_b": current_key[1], "rows": current_rows}


def grid_from_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array(sorted({round(row["x_mm"], 8) for row in rows}), dtype=np.float64)
    ys = np.array(sorted({round(row["y_mm"], 8) for row in rows}), dtype=np.float64)
    x_index = {value: idx for idx, value in enumerate(xs)}
    y_index = {value: idx for idx, value in enumerate(ys)}
    u = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    v = np.full_like(u, np.nan)
    for row in rows:
        col = x_index[round(row["x_mm"], 8)]
        line = y_index[round(row["y_mm"], 8)]
        u[line, col] = row["u_mm_s"]
        v[line, col] = row["v_mm_s"]
    return xs, ys, u, v


def nearest_fill(field: np.ndarray) -> np.ndarray:
    if np.isfinite(field).all():
        return field
    out = field.copy()
    rows, cols = out.shape
    for _ in range(rows + cols):
        missing = ~np.isfinite(out)
        if not missing.any():
            break
        updated = out.copy()
        for r in range(rows):
            for c in range(cols):
                if not missing[r, c]:
                    continue
                vals = []
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < rows and 0 <= cc < cols and np.isfinite(out[rr, cc]):
                        vals.append(out[rr, cc])
                if vals:
                    updated[r, c] = float(np.mean(vals))
        if np.array_equal(np.isfinite(updated), np.isfinite(out)):
            break
        out = updated
    return np.nan_to_num(out, nan=0.0)


def compute_vorticity(xs_mm: np.ndarray, ys_mm_down: np.ndarray, u_mm_s: np.ndarray, v_down_mm_s: np.ndarray) -> np.ndarray:
    u = nearest_fill(u_mm_s)
    v_down = nearest_fill(v_down_mm_s)
    dx = float(np.median(np.diff(xs_mm))) if len(xs_mm) > 1 else 1.0
    dy_down = float(np.median(np.diff(ys_mm_down))) if len(ys_mm_down) > 1 else 1.0
    du_dy_down = np.gradient(u, dy_down, axis=0)
    dv_down_dx = np.gradient(v_down, dx, axis=1)
    # Physical convention: x right, y up. CSV uses image y down, so omega_z = du/dy_down - dv_down/dx.
    return du_dy_down - dv_down_dx


def filter_vorticity(omega: np.ndarray, settings: dict) -> np.ndarray:
    out = omega.astype(np.float32).copy()
    gaussian_sigma = float(settings.get("gaussian_sigma", 0.0))
    if gaussian_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), gaussian_sigma)

    threshold = settings.get("threshold_1_s")
    if threshold is not None:
        threshold = float(threshold)
        out = np.where(np.abs(out) >= threshold, out, 0.0).astype(np.float32)

    percentile = settings.get("threshold_percentile")
    if percentile is not None:
        percentile = float(percentile)
        finite = np.abs(out[np.isfinite(out)])
        if finite.size:
            local_threshold = float(np.percentile(finite, percentile))
            out = np.where(np.abs(out) >= local_threshold, out, 0.0).astype(np.float32)

    return out


def robust_limit(values: list[np.ndarray], percentile: float) -> float:
    if not values:
        return 1.0
    flat = np.concatenate([np.abs(value[np.isfinite(value)]).ravel() for value in values])
    if flat.size == 0:
        return 1.0
    limit = float(np.percentile(flat, percentile))
    return max(limit, 1e-6)


def colorize_signed(
    field: np.ndarray,
    limit: float,
    alpha: float,
    transparent_zero: bool = True,
    alpha_by_magnitude: bool = False,
) -> np.ndarray:
    norm = np.clip((field / limit + 1.0) * 0.5, 0.0, 1.0)
    rgba = COOLWARM(norm)
    bgr = (rgba[:, :, 2::-1] * 255).astype(np.uint8)
    if alpha_by_magnitude:
        strength = np.sqrt(np.clip(np.abs(field) / max(limit, 1e-12), 0.0, 1.0))
        overlay_alpha = np.clip(alpha * 255.0 * strength, 0, 255).astype(np.uint8)
    else:
        overlay_alpha = np.full(field.shape, int(round(alpha * 255)), dtype=np.uint8)
    if transparent_zero:
        overlay_alpha[np.abs(field) <= 0] = 0
    return np.dstack([bgr, overlay_alpha])


def alpha_blend_roi(frame: np.ndarray, roi_bgra: np.ndarray, roi: list[int]) -> None:
    x, y, w, h = [int(v) for v in roi]
    roi_frame = frame[y : y + h, x : x + w]
    alpha = roi_bgra[:, :, 3:4].astype(np.float32) / 255.0
    color = roi_bgra[:, :, :3].astype(np.float32)
    blended = color * alpha + roi_frame.astype(np.float32) * (1.0 - alpha)
    frame[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)


def apply_white_roi_background(frame: np.ndarray, roi: list[int], alpha: float) -> None:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return
    x, y, w, h = [int(v) for v in roi]
    roi_frame = frame[y : y + h, x : x + w]
    white = np.full_like(roi_frame, 255)
    frame[y : y + h, x : x + w] = cv2.addWeighted(white, alpha, roi_frame, 1.0 - alpha, 0.0)


def draw_colorbar(frame: np.ndarray, limit: float) -> None:
    h, w = frame.shape[:2]
    x0 = max(8, w - 58)
    y0 = 90
    bar_w = 18
    bar_h = min(280, max(140, h - 220))
    vals = np.linspace(limit, -limit, bar_h)
    colors = (COOLWARM((vals / limit + 1) * 0.5)[:, 2::-1] * 255).astype(np.uint8)
    for i, color in enumerate(colors):
        cv2.line(frame, (x0, y0 + i), (x0 + bar_w, y0 + i), tuple(int(v) for v in color), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (235, 235, 235), 1)
    cv2.putText(frame, f"+{limit:.2g}", (x0 - 12, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    cv2.putText(frame, "0", (x0 + bar_w + 4, y0 + bar_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    cv2.putText(frame, f"-{limit:.2g}", (x0 - 12, y0 + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    cv2.putText(frame, "omega [1/s]", (x0 - 96, y0 + bar_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)


def draw_frame(
    frame: np.ndarray,
    omega: np.ndarray,
    grid_x_px: np.ndarray,
    grid_y_px: np.ndarray,
    roi: list[int],
    frame_a: int,
    frame_b: int,
    time_s: float,
    limit: float,
    alpha: float,
    transparent_zero: bool,
    alpha_by_magnitude: bool,
    white_background_alpha: float,
) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = [int(v) for v in roi]
    apply_white_roi_background(out, roi, white_background_alpha)
    omega_img = cv2.resize(omega.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = colorize_signed(
        omega_img,
        limit,
        alpha,
        transparent_zero=transparent_zero,
        alpha_by_magnitude=alpha_by_magnitude,
    )
    alpha_blend_roi(out, heat, roi)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)

    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (470, 72), (0, 0, 0), -1)
    out = cv2.addWeighted(panel, 0.45, out, 0.55, 0)
    cv2.putText(out, f"Vorticidad frame {frame_a}->{frame_b}  t={time_s:.3f}s", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "convencion fisica: x derecha, y arriba", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    draw_colorbar(out, limit)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Calcula vorticidad desde vectores PIV y genera video.")
    parser.add_argument("--summary", type=Path, default=Path("resultados/ensayo_placa_2/summary.json"))
    parser.add_argument("--vectors", type=Path, default=Path("resultados/ensayo_placa_2/vectors_mm_s.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("resultados/ensayo_placa_2"))
    parser.add_argument("--file", default="vorticity_overlay.mp4")
    parser.add_argument("--fps", type=float, default=None, help="FPS de reproducción del video reconstruido.")
    parser.add_argument("--csv", default="vorticity.csv")
    parser.add_argument("--no-csv", action="store_true", help="No genera el CSV auxiliar de vorticidad.")
    parser.add_argument("--alpha", type=float, default=0.68)
    parser.add_argument("--limit", type=float, default=None, help="Escala de color simetrica en 1/s.")
    parser.add_argument("--percentile", type=float, default=98.0)
    parser.add_argument("--gaussian-sigma", type=float, default=0.0, help="Suavizado espacial previo en celdas de grilla.")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral absoluto de vorticidad en 1/s.")
    parser.add_argument("--threshold-percentile", type=float, default=None, help="Umbral por percentil local de |omega| en cada frame.")
    parser.add_argument("--show-zero", action="store_true", help="Muestra tambien zonas filtradas como color neutro.")
    parser.add_argument(
        "--alpha-by-magnitude",
        action="store_true",
        help="Escala la opacidad con la raiz de |omega|/limite global.",
    )
    parser.add_argument(
        "--white-background-alpha",
        type=float,
        default=0.0,
        help="Opacidad del velo blanco dentro del ROI, entre 0 y 1.",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    roi = summary["roi"]
    fps = (
        float(args.fps)
        if args.fps is not None
        else float(summary["fps"]) / max(1, int(summary.get("stride", 1)))
    )
    if fps <= 0:
        raise RuntimeError("El FPS de salida debe ser mayor que cero.")
    video_path = Path(summary["video"])
    frames = read_vectors(args.vectors)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    computed = []
    filter_settings = {
        "gaussian_sigma": args.gaussian_sigma,
        "threshold_1_s": args.threshold,
        "threshold_percentile": args.threshold_percentile,
    }
    for item in frames:
        xs_mm, ys_mm_down, u, v_down = grid_from_rows(item["rows"])
        omega = compute_vorticity(xs_mm, ys_mm_down, u, v_down)
        omega = filter_vorticity(omega, filter_settings)
        grid_x_px = np.array(sorted({row["x_px"] for row in item["rows"]}), dtype=np.float64)
        grid_y_px = np.array(sorted({row["y_px"] for row in item["rows"]}), dtype=np.float64)
        computed.append({**item, "omega": omega, "grid_x_px": grid_x_px, "grid_y_px": grid_y_px})

    limit = args.limit if args.limit is not None else robust_limit([item["omega"] for item in computed], args.percentile)

    vorticity_csv_path = None
    if not args.no_csv:
        vorticity_csv_path = args.out_dir / args.csv
        with vorticity_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame_a", "frame_b", "x_px", "y_px", "omega_1_s"])
            for item in computed:
                omega = item["omega"]
                for row_idx, y_px in enumerate(item["grid_y_px"]):
                    for col_idx, x_px in enumerate(item["grid_x_px"]):
                        writer.writerow([item["frame_a"], item["frame_b"], float(x_px), float(y_px), float(omega[row_idx, col_idx])])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.out_dir / args.file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("No se pudo crear el video de vorticidad.")

    for idx, item in enumerate(computed, start=1):
        print(f"Renderizando vorticidad {idx}/{len(computed)}: frame {item['frame_a']} -> {item['frame_b']}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame_a"])
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer frame {item['frame_a']}")
        out = draw_frame(
            frame,
            item["omega"],
            item["grid_x_px"],
            item["grid_y_px"],
            roi,
            item["frame_a"],
            item["frame_b"],
            item["frame_a"] / float(summary["fps"]),
            limit,
            args.alpha,
            not args.show_zero,
            args.alpha_by_magnitude,
            float(np.clip(args.white_background_alpha, 0.0, 1.0)),
        )
        writer.write(out)

    cap.release()
    writer.release()

    out_summary = {
        "source_vectors": str(args.vectors),
        "video": str(args.out_dir / args.file),
        "csv": str(vorticity_csv_path) if vorticity_csv_path else None,
        "frames": len(computed),
        "fps": fps,
        "roi": roi,
        "vorticity_units": "1/s",
        "sign_convention": "x right, y up; positive omega_z follows right-hand rule",
        "color_limit_1_s": limit,
        "color_percentile": args.percentile if args.limit is None else None,
        "alpha": args.alpha,
        "filter": filter_settings,
        "transparent_zero": not args.show_zero,
        "alpha_by_magnitude": args.alpha_by_magnitude,
        "white_background_alpha": float(np.clip(args.white_background_alpha, 0.0, 1.0)),
    }
    with (args.out_dir / "vorticity_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(out_summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
