"""Finaliza campos PIV desde checkpoints compactos y genera los dos videos finales."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np

import procesar_piv as piv
from generar_video_vorticidad import compute_vorticity, filter_vorticity


TURBO_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0, :]
COOLWARM_RGB = matplotlib.colormaps["coolwarm"](np.linspace(0.0, 1.0, 256))[:, :3]
COOLWARM_BGR = np.rint(COOLWARM_RGB[:, ::-1] * 255.0).astype(np.uint8)


def color_from_lut(value: float, limit: float, lut: np.ndarray) -> tuple[int, int, int]:
    index = int(np.clip(round(255.0 * value / max(limit, 1e-12)), 0, 255))
    color = lut[index]
    return int(color[0]), int(color[1]), int(color[2])


def white_roi(frame: np.ndarray, roi: list[int], alpha: float) -> None:
    x, y, width, height = [int(value) for value in roi]
    source = frame[y : y + height, x : x + width]
    frame[y : y + height, x : x + width] = cv2.addWeighted(
        np.full_like(source, 255), float(alpha), source, 1.0 - float(alpha), 0.0
    )


def mask_plate_visual(frame: np.ndarray, polygon: np.ndarray | None) -> None:
    if polygon is None:
        return
    cv2.fillPoly(frame, [polygon], (245, 245, 245), cv2.LINE_AA)
    cv2.polylines(frame, [polygon], True, (105, 105, 105), 1, cv2.LINE_AA)


def draw_speed_colorbar(frame: np.ndarray, limit: float) -> None:
    height, width = frame.shape[:2]
    x0, y0 = width - 58, 92
    bar_width, bar_height = 18, min(270, height - 220)
    panel = frame.copy()
    cv2.rectangle(panel, (x0 - 118, y0 - 34), (width - 5, y0 + bar_height + 55), (255, 255, 255), -1)
    frame[:] = cv2.addWeighted(panel, 0.88, frame, 0.12, 0.0)
    for row in range(bar_height):
        fraction = 1.0 - row / max(1, bar_height - 1)
        cv2.line(frame, (x0, y0 + row), (x0 + bar_width, y0 + row), color_from_lut(fraction, 1.0, TURBO_LUT), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_width, y0 + bar_height), (40, 40, 40), 1)
    cv2.putText(frame, f"{limit:.3g}", (x0 - 16, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(frame, "0", (x0 + 4, y0 + bar_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(frame, "|U| [mm/s]", (x0 - 105, y0 + bar_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)


def draw_vorticity_colorbar(frame: np.ndarray, limit: float) -> None:
    height, width = frame.shape[:2]
    x0, y0 = width - 58, 92
    bar_width, bar_height = 18, min(270, height - 220)
    panel = frame.copy()
    cv2.rectangle(panel, (x0 - 118, y0 - 34), (width - 5, y0 + bar_height + 55), (255, 255, 255), -1)
    frame[:] = cv2.addWeighted(panel, 0.88, frame, 0.12, 0.0)
    for row in range(bar_height):
        signed = 1.0 - 2.0 * row / max(1, bar_height - 1)
        index = int(np.clip(round(255.0 * (signed + 1.0) * 0.5), 0, 255))
        color = COOLWARM_BGR[index]
        cv2.line(frame, (x0, y0 + row), (x0 + bar_width, y0 + row), tuple(int(v) for v in color), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_width, y0 + bar_height), (40, 40, 40), 1)
    cv2.putText(frame, f"+{limit:.3g}", (x0 - 25, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(frame, "0", (x0 + bar_width + 5, y0 + bar_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(frame, f"-{limit:.3g}", (x0 - 25, y0 + bar_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(frame, "omega_z [1/s]", (x0 - 110, y0 + bar_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)


def info_panel(frame: np.ndarray, line1: str, line2: str) -> None:
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (710, 78), (255, 255, 255), -1)
    frame[:] = cv2.addWeighted(panel, 0.78, frame, 0.22, 0.0)
    cv2.putText(frame, line1, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)


def render_velocity(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    speed_limit: float,
    roi: list[int],
    plate_polygon: np.ndarray | None,
    frame_a: int,
    frame_b: int,
    time_s: float,
    fixed_length_px: float,
    method_label: str,
) -> np.ndarray:
    out = frame.copy()
    white_roi(out, roi, 0.62)
    rows, cols = x.shape
    for row in range(rows):
        for col in range(cols):
            if not valid[row, col]:
                continue
            du, dv = float(u[row, col]), float(v[row, col])
            speed = float(np.hypot(du, dv))
            if not np.isfinite(speed) or speed <= 1e-10:
                continue
            start = (int(round(float(x[row, col]))), int(round(float(y[row, col]))))
            end = (
                int(round(start[0] + fixed_length_px * du / speed)),
                int(round(start[1] + fixed_length_px * dv / speed)),
            )
            cv2.arrowedLine(
                out,
                start,
                end,
                color_from_lut(speed, speed_limit, TURBO_LUT),
                1,
                cv2.LINE_AA,
                tipLength=0.34,
            )
    mask_plate_visual(out, plate_polygon)
    draw_speed_colorbar(out, speed_limit)
    info_panel(
        out,
        f"PIV FFT frame {frame_a}->{frame_b}  t={time_s:.3f} s",
        f"{method_label}; validos={100.0 * float(np.mean(valid)):.1f} %; flechas fijas",
    )
    return out


def interpolate_field_to_roi(
    field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    roi: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    rx, ry, width, height = [int(value) for value in roi]
    step_x = float(np.median(np.diff(x_grid[0])))
    step_y = float(np.median(np.diff(y_grid[:, 0])))
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32) + rx, np.arange(height, dtype=np.float32) + ry)
    map_x = (xx - float(x_grid[0, 0])) / step_x
    map_y = (yy - float(y_grid[0, 0])) / step_y
    finite = np.isfinite(field).astype(np.uint8)
    interpolated = cv2.remap(
        np.nan_to_num(field, nan=0.0).astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    coverage = cv2.remap(finite, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return interpolated, coverage.astype(bool)


def render_vorticity(
    frame: np.ndarray,
    omega: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    limit: float,
    roi: list[int],
    plate_polygon: np.ndarray | None,
    frame_a: int,
    frame_b: int,
    time_s: float,
    vorticity_sigma: float = 1.25,
    vorticity_threshold: float | None = None,
    color_scale_label: str = "escala global max",
) -> np.ndarray:
    out = frame.copy()
    white_roi(out, roi, 0.62)
    rx, ry, width, height = [int(value) for value in roi]
    field, coverage = interpolate_field_to_roi(omega, x_grid, y_grid, roi)
    normalized = np.clip((field / max(limit, 1e-12) + 1.0) * 0.5, 0.0, 1.0)
    heat = COOLWARM_BGR[np.rint(normalized * 255.0).astype(np.uint8)]
    source = out[ry : ry + height, rx : rx + width].astype(np.float32)
    alpha = (0.78 * coverage.astype(np.float32))[:, :, None]
    out[ry : ry + height, rx : rx + width] = np.clip(heat * alpha + source * (1.0 - alpha), 0, 255).astype(np.uint8)
    mask_plate_visual(out, plate_polygon)
    draw_vorticity_colorbar(out, limit)
    info_panel(
        out,
        f"Vorticidad frame {frame_a}->{frame_b}  t={time_s:.3f} s",
        (
            "omega_z=dv/dx-du/dy; "
            f"gauss sigma={vorticity_sigma:g} nodos; "
            + (f"umbral |omega|<{vorticity_threshold:g} s^-1; " if vorticity_threshold is not None else "sin umbral; ")
            + color_scale_label
        ),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--vorticity-file", required=True)
    parser.add_argument("--report-file", default="piv_fft_resumen.json")
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--output-duration", type=float, default=None)
    parser.add_argument("--vorticity-sigma", type=float, default=3.0)
    parser.add_argument("--vorticity-threshold", type=float, default=1.0)
    parser.add_argument(
        "--tail-frames-dir",
        type=Path,
        default=None,
        help="Directorio opcional con PNG frame_XXXXXX.png para el tramo final si OpenCV no decodifica el MP4.",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    pair_count = int(summary["pairs_processed"])
    results = []
    for pair_idx in range(pair_count):
        path = args.checkpoints / f"pair_{pair_idx:06d}.pkl"
        with path.open("rb") as fh:
            results.append(pickle.load(fh))
    piv.apply_temporal_filter(results, summary["temporal_filter"])

    frame_a = np.array([item["frame_a"] for item in results], dtype=np.int32)
    frame_b = np.array([item["frame_b"] for item in results], dtype=np.int32)
    x = results[0]["x"].astype(np.float32)
    y = results[0]["y"].astype(np.float32)
    u_px = np.stack([item["u_out"] for item in results]).astype(np.float32)
    v_px = np.stack([item["v_out"] for item in results]).astype(np.float32)
    corr = np.stack([item["corr"] for item in results]).astype(np.float32)
    valid = np.stack([item["valid_out"] for item in results]).astype(bool)
    dt = float(summary["dt_s"])
    mm_per_px = float(summary["calibration"]["mm_per_px"])
    u_mm_s = u_px * mm_per_px / dt
    v_mm_s = v_px * mm_per_px / dt
    speed = np.hypot(u_mm_s, v_mm_s)
    speed_limit = float(np.nanmax(np.where(valid, speed, np.nan)))
    pass_windows = [int(item["window"]) for item in summary["passes"]]
    passes_label = "->".join(str(window) for window in pass_windows)
    final_pass = summary["passes"][-1]
    final_step_px = max(1, int(round(float(final_pass["window"]) * (1.0 - float(final_pass["overlap"])))))
    fixed_arrow_length_px = 0.75 * final_step_px
    temporal_radius = int(summary["temporal_filter"].get("radius", 0))
    temporal_measurements = 2 * temporal_radius + 1
    frame_delta = int(frame_b[0] - frame_a[0])
    method_label = (
        f"{passes_label} px; delta={frame_delta}; "
        f"mediana {temporal_measurements}"
    )

    mask_polygons = summary.get("mask_polygons", [])
    plate_polygon = (
        np.asarray(mask_polygons[0], dtype=np.int32)
        if mask_polygons
        else None
    )
    plate_grid_mask = np.zeros_like(x, dtype=np.uint8)
    if plate_polygon is not None:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                if cv2.pointPolygonTest(plate_polygon, (float(x[row, col]), float(y[row, col])), False) >= 0:
                    plate_grid_mask[row, col] = 1
    plate_grid_mask = plate_grid_mask.astype(bool)

    xs_mm = x[0].astype(np.float64) * mm_per_px
    ys_mm_down = y[:, 0].astype(np.float64) * mm_per_px
    omega_fields = []
    omega_limit = 0.0
    for idx in range(pair_count):
        omega = compute_vorticity(xs_mm, ys_mm_down, u_mm_s[idx], v_mm_s[idx])
        omega = filter_vorticity(
            omega,
            {
                "gaussian_sigma": float(args.vorticity_sigma),
                "threshold_1_s": float(args.vorticity_threshold),
            },
        )
        omega = np.where(plate_grid_mask, np.nan, omega).astype(np.float32)
        omega_limit = max(omega_limit, float(np.nanmax(np.abs(omega))))
        omega_fields.append(omega)
    omega_stack = np.stack(omega_fields).astype(np.float32)

    args.compact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.compact,
        frame_a=frame_a,
        frame_b=frame_b,
        x_px=x,
        y_px=y,
        u_mm_s=u_mm_s,
        v_mm_s=v_mm_s,
        corr=corr,
        valid=valid,
        omega_1_s=omega_stack,
        plate_polygon_px=plate_polygon,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    velocity_path = args.out_dir / args.velocity_file
    vorticity_path = args.out_dir / args.vorticity_file
    size = (int(summary["width"]), int(summary["height"]))
    output_fps = float(args.output_fps if args.output_fps is not None else summary.get("output_fps", 30.0))
    output_duration = args.output_duration if args.output_duration is not None else summary.get("output_duration_s")
    render_frames = int(round(float(output_duration) * output_fps)) if output_duration else pair_count
    render_frames = max(1, render_frames)
    velocity_writer = cv2.VideoWriter(str(velocity_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    vorticity_writer = cv2.VideoWriter(str(vorticity_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    if not velocity_writer.isOpened() or not vorticity_writer.isOpened():
        raise RuntimeError("No se pudieron abrir los escritores de video.")

    cap = cv2.VideoCapture(str(summary["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {summary['video']}")
    current_frame = -1
    source_frame = None
    tail_frames_dir = args.tail_frames_dir
    for render_idx in range(render_frames):
        if render_frames == 1:
            idx = 0
        else:
            idx = int(round(render_idx * (pair_count - 1) / (render_frames - 1)))
        idx = min(pair_count - 1, max(0, idx))
        target = int(frame_a[idx])
        tail_path = tail_frames_dir / f"frame_{target:06d}.png" if tail_frames_dir is not None else None
        if tail_path is not None and tail_path.is_file():
            source_frame = cv2.imread(str(tail_path), cv2.IMREAD_COLOR)
            if source_frame is None:
                raise RuntimeError(f"No se pudo leer el frame PNG {tail_path}.")
            current_frame = target
        else:
            while current_frame < target:
                ok, source_frame = cap.read()
                current_frame += 1
                if not ok:
                    raise RuntimeError(f"No se pudo decodificar secuencialmente el frame {current_frame}.")
        velocity_writer.write(
            render_velocity(
                source_frame,
                x,
                y,
                u_mm_s[idx],
                v_mm_s[idx],
                valid[idx],
                speed_limit,
                summary["roi"],
                plate_polygon,
                int(frame_a[idx]),
                int(frame_b[idx]),
                float(frame_a[idx]) / float(summary["fps"]),
                fixed_arrow_length_px,
                method_label,
            )
        )
        vorticity_writer.write(
            render_vorticity(
                source_frame,
                omega_stack[idx],
                x,
                y,
                omega_limit,
                summary["roi"],
                plate_polygon,
                int(frame_a[idx]),
                int(frame_b[idx]),
                float(frame_a[idx]) / float(summary["fps"]),
            )
        )
        if (render_idx + 1) % 50 == 0 or render_idx + 1 == render_frames:
            print(f"Renderizados {render_idx + 1}/{render_frames}", flush=True)

    cap.release()
    velocity_writer.release()
    vorticity_writer.release()

    output_summary = {
        "source_summary": str(args.summary),
        "source_checkpoints": str(args.checkpoints),
        "compact_fields": str(args.compact),
        "velocity_video": str(velocity_path),
        "vorticity_video": str(vorticity_path),
        "frames": render_frames,
        "source_fields": pair_count,
        "video_fps": output_fps,
        "output_duration_s": render_frames / output_fps,
        "speed_color_limit_global_max_mm_s": speed_limit,
        "vorticity_color_limit_global_max_1_s": omega_limit,
        "passes": summary["passes"],
        "final_grid_step_px": final_step_px,
        "fixed_arrow_length_px": fixed_arrow_length_px,
        "white_background_alpha": 0.62,
        "vorticity_gaussian_sigma_grid_nodes": float(args.vorticity_sigma),
        "vorticity_threshold": float(args.vorticity_threshold),
        "valid_fraction": float(np.mean(valid)),
        "plate_mask_polygon_px": plate_polygon.tolist() if plate_polygon is not None else None,
    }
    (args.out_dir / args.report_file).write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
