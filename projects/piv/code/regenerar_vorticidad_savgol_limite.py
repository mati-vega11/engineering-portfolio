"""Regenera solo el video de vorticidad Savitzky--Golay con rango fijo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from finalizar_piv_video_optimizado import compute_vorticity, filter_vorticity, render_vorticity
from savgol_piv_video_desde_compacto import mask_grid, savgol_2d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vorticity-file", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--vorticity-sigma", type=float, default=3.0)
    parser.add_argument(
        "--vorticity-threshold",
        type=float,
        default=None,
        help="Umbral absoluto en s^-1; valores menores se anulan.",
    )
    parser.add_argument("--color-limit", type=float, default=1.0)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--output-duration", type=float, default=None)
    args = parser.parse_args()
    if args.color_limit <= 0:
        raise ValueError("El límite cromático debe ser positivo.")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    fields = np.load(args.compact, allow_pickle=True)
    frame_a = np.asarray(fields["frame_a"], dtype=np.int32)
    frame_b = np.asarray(fields["frame_b"], dtype=np.int32)
    x = np.asarray(fields["x_px"], dtype=np.float32)
    y = np.asarray(fields["y_px"], dtype=np.float32)
    u_mm_s = np.asarray(fields["u_mm_s"], dtype=np.float32)
    v_mm_s = np.asarray(fields["v_mm_s"], dtype=np.float32)
    corr = np.asarray(fields["corr"], dtype=np.float32)
    valid = np.asarray(fields["valid"], dtype=bool)
    min_corr = float(summary.get("min_corr", 0.18))
    valid &= np.isfinite(u_mm_s) & np.isfinite(v_mm_s) & np.isfinite(corr) & (corr >= min_corr)

    mask_polygons = summary.get("mask_polygons", [])
    plate_polygon = np.asarray(mask_polygons[0], dtype=np.int32) if mask_polygons else None
    plate_mask = mask_grid(x, y, plate_polygon)
    valid &= ~plate_mask[None, :, :]

    u_smooth = np.empty_like(u_mm_s, dtype=np.float32)
    v_smooth = np.empty_like(v_mm_s, dtype=np.float32)
    for idx in range(u_mm_s.shape[0]):
        u_smooth[idx] = savgol_2d(u_mm_s[idx], valid[idx], args.window, args.polyorder)
        v_smooth[idx] = savgol_2d(v_mm_s[idx], valid[idx], args.window, args.polyorder)
        if (idx + 1) % 100 == 0 or idx + 1 == u_mm_s.shape[0]:
            print(f"Suavizado {idx + 1}/{u_mm_s.shape[0]}", flush=True)

    mm_per_px = float(summary["calibration"]["mm_per_px"])
    xs_mm = x[0].astype(np.float64) * mm_per_px
    ys_mm_down = y[:, 0].astype(np.float64) * mm_per_px
    omega_fields: list[np.ndarray] = []
    for idx in range(u_smooth.shape[0]):
        omega = compute_vorticity(xs_mm, ys_mm_down, u_smooth[idx], v_smooth[idx])
        omega = filter_vorticity(
            omega,
            {
                "gaussian_sigma": float(args.vorticity_sigma),
                "threshold_1_s": args.vorticity_threshold,
            },
        )
        omega_fields.append(np.where(plate_mask, np.nan, omega).astype(np.float32))
    omega_stack = np.stack(omega_fields).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vorticity_path = args.out_dir / args.vorticity_file
    size = (int(summary["width"]), int(summary["height"]))
    output_fps = float(args.output_fps if args.output_fps is not None else summary.get("output_fps", 30.0))
    output_duration = args.output_duration if args.output_duration is not None else summary.get("output_duration_s")
    render_frames = int(round(float(output_duration) * output_fps)) if output_duration else len(frame_a)
    render_frames = max(1, render_frames)
    writer = cv2.VideoWriter(str(vorticity_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    if not writer.isOpened():
        raise RuntimeError("No se pudo abrir el escritor de vorticidad.")

    cap = cv2.VideoCapture(str(summary["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {summary['video']}")
    current_frame = -1
    source_frame: np.ndarray | None = None
    for render_idx in range(render_frames):
        idx = 0 if render_frames == 1 else int(round(render_idx * (len(frame_a) - 1) / (render_frames - 1)))
        idx = min(len(frame_a) - 1, max(0, idx))
        target = int(frame_a[idx])
        while current_frame < target:
            ok, source_frame = cap.read()
            current_frame += 1
            if not ok or source_frame is None:
                raise RuntimeError(f"No se pudo decodificar secuencialmente el frame {current_frame}.")
        writer.write(
            render_vorticity(
                source_frame,
                omega_stack[idx],
                x,
                y,
                float(args.color_limit),
                summary["roi"],
                plate_polygon,
                int(frame_a[idx]),
                int(frame_b[idx]),
                float(frame_a[idx]) / float(summary["fps"]),
                vorticity_sigma=float(args.vorticity_sigma),
                vorticity_threshold=None,
                color_scale_label=(
                    f"escala fija +/-{args.color_limit:g} s^-1; "
                    + (
                        f"umbral |omega|>={args.vorticity_threshold:g} s^-1"
                        if args.vorticity_threshold is not None
                        else "sin umbral"
                    )
                ),
            )
        )
        if (render_idx + 1) % 50 == 0 or render_idx + 1 == render_frames:
            print(f"Renderizados {render_idx + 1}/{render_frames}", flush=True)

    cap.release()
    writer.release()

    report_path = args.out_dir / args.report_file
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report.update(
        {
            "method": "Savitzky-Golay espacial separable",
            "vorticity_video": str(vorticity_path),
            "vorticity_color_limit_mode": "fixed_symmetric",
            "vorticity_color_limit_abs_1_s": float(args.color_limit),
            "vorticity_color_saturation": "values <= -limit blue; values >= +limit red",
            "vorticity_gaussian_sigma_grid_nodes": float(args.vorticity_sigma),
            "vorticity_threshold": (
                float(args.vorticity_threshold) if args.vorticity_threshold is not None else None
            ),
            "output_fps": output_fps,
            "output_duration_s": render_frames / output_fps,
        }
    )
    report.setdefault("parameters", {})
    report["parameters"].update(
        {
            "window_rows_x_cols": f"{args.window}x{args.window}",
            "polynomial_order": int(args.polyorder),
            "vorticity_color_limit_abs_1_s": float(args.color_limit),
            "vorticity_color_limit_mode": "fixed_symmetric",
            "vorticity_threshold_abs_1_s": (
                float(args.vorticity_threshold) if args.vorticity_threshold is not None else None
            ),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
