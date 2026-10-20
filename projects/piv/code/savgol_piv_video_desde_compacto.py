"""Suaviza campos PIV con Savitzky--Golay y genera velocidad/vorticidad.

El NPZ de entrada contiene los campos PIV con el filtro temporal ya aplicado.
Se suavizan espacialmente u y v con un filtro Savitzky--Golay separable de
ventana 5x5 y polinomio de orden 2. Los vectores originalmente inválidos y la
placa se restauran como inválidos después del suavizado.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter

from finalizar_piv_video_optimizado import compute_vorticity, filter_vorticity, render_velocity, render_vorticity


def nearest_fill(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rellena huecos con el valor válido más cercano para evitar NaN en SG."""
    if bool(np.all(valid)):
        return field.astype(np.float32, copy=True)
    if not bool(np.any(valid)):
        return np.zeros_like(field, dtype=np.float32)
    invalid = ~valid
    _, indices = distance_transform_edt(invalid, return_distances=True, return_indices=True)
    filled = field[tuple(indices)]
    return np.asarray(filled, dtype=np.float32)


def savgol_2d(field: np.ndarray, valid: np.ndarray, window: int, order: int) -> np.ndarray:
    """Aplica Savitzky--Golay separable en filas y columnas."""
    filled = nearest_fill(field, valid)
    filtered_y = savgol_filter(filled, window_length=window, polyorder=order, axis=0, mode="interp")
    filtered_xy = savgol_filter(filtered_y, window_length=window, polyorder=order, axis=1, mode="interp")
    return np.where(valid, filtered_xy, np.nan).astype(np.float32)


def mask_grid(x: np.ndarray, y: np.ndarray, polygon: np.ndarray | None) -> np.ndarray:
    if polygon is None:
        return np.zeros_like(x, dtype=bool)
    out = np.zeros_like(x, dtype=bool)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            out[row, col] = cv2.pointPolygonTest(
                polygon, (float(x[row, col]), float(y[row, col])), False
            ) >= 0
    return out


def finite_max_abs(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(np.max(finite)) if finite.size else 0.0, 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--vorticity-file", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--vorticity-sigma", type=float, default=3.0)
    parser.add_argument(
        "--vorticity-color-limit",
        type=float,
        default=None,
        help="Limite simetrico fijo +/- en s^-1; por defecto usa el maximo global.",
    )
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--output-duration", type=float, default=None)
    args = parser.parse_args()
    if args.window % 2 != 1 or args.polyorder >= args.window:
        raise ValueError("La ventana debe ser impar y mayor que el orden del polinomio.")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    fields = np.load(args.compact, allow_pickle=True)
    frame_a = np.asarray(fields["frame_a"], dtype=np.int32)
    frame_b = np.asarray(fields["frame_b"], dtype=np.int32)
    # Algunas versiones del compacto guardan x/y una vez (2-D) y otras
    # repiten la grilla para cada campo (3-D). Para renderizar se usa siempre
    # la grilla del primer campo.
    x_raw = np.asarray(fields["x_px"], dtype=np.float32)
    y_raw = np.asarray(fields["y_px"], dtype=np.float32)
    x = x_raw[0] if x_raw.ndim == 3 else x_raw
    y = y_raw[0] if y_raw.ndim == 3 else y_raw
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
            {"gaussian_sigma": float(args.vorticity_sigma), "threshold_1_s": None},
        )
        omega_fields.append(np.where(plate_mask, np.nan, omega).astype(np.float32))
    omega_stack = np.stack(omega_fields).astype(np.float32)

    speed_limit = finite_max_abs(np.hypot(u_smooth, v_smooth))
    omega_global_max = finite_max_abs(omega_stack)
    omega_limit = (
        float(args.vorticity_color_limit)
        if args.vorticity_color_limit is not None
        else omega_global_max
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    velocity_path = args.out_dir / args.velocity_file
    vorticity_path = args.out_dir / args.vorticity_file
    size = (int(summary["width"]), int(summary["height"]))
    output_fps = float(args.output_fps if args.output_fps is not None else summary.get("output_fps", 30.0))
    output_duration = args.output_duration if args.output_duration is not None else summary.get("output_duration_s")
    render_frames = int(round(float(output_duration) * output_fps)) if output_duration else len(frame_a)
    render_frames = max(1, render_frames)
    velocity_writer = cv2.VideoWriter(str(velocity_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    vorticity_writer = cv2.VideoWriter(str(vorticity_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    if not velocity_writer.isOpened() or not vorticity_writer.isOpened():
        raise RuntimeError("No se pudieron abrir los escritores de video.")

    cap = cv2.VideoCapture(str(summary["video"]))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {summary['video']}")
    current_frame = -1
    source_frame: np.ndarray | None = None
    final_pass = summary["passes"][-1]
    final_step_px = max(1, int(round(float(final_pass["window"]) * (1.0 - float(final_pass["overlap"])))) )
    fixed_arrow_length_px = 0.75 * final_step_px
    radius = int(summary.get("temporal_filter", {}).get("radius", 0))
    temporal_measurements = 2 * radius + 1
    method_label = (
        f"Savitzky-Golay {args.window}x{args.window}, orden {args.polyorder}; "
        f"mediana temporal {temporal_measurements}"
    )
    for render_idx in range(render_frames):
        idx = 0 if render_frames == 1 else int(round(render_idx * (len(frame_a) - 1) / (render_frames - 1)))
        idx = min(len(frame_a) - 1, max(0, idx))
        target = int(frame_a[idx])
        while current_frame < target:
            ok, source_frame = cap.read()
            current_frame += 1
            if not ok or source_frame is None:
                raise RuntimeError(f"No se pudo decodificar secuencialmente el frame {current_frame}.")
        velocity_writer.write(
            render_velocity(
                source_frame,
                x,
                y,
                u_smooth[idx],
                v_smooth[idx],
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
                vorticity_sigma=float(args.vorticity_sigma),
                vorticity_threshold=None,
                color_scale_label=(
                    f"escala fija +/-{omega_limit:g} s^-1 (saturacion)"
                    if args.vorticity_color_limit is not None
                    else "escala global max |omega|"
                ),
            )
        )
        if (render_idx + 1) % 50 == 0 or render_idx + 1 == render_frames:
            print(f"Renderizados {render_idx + 1}/{render_frames}", flush=True)

    cap.release()
    velocity_writer.release()
    vorticity_writer.release()

    report = {
        "method": "Savitzky-Golay espacial separable",
        "source_summary": str(args.summary),
        "source_compact_fields": str(args.compact),
        "note": "Aplicado sobre los campos PIV ya filtrados temporalmente; se relleneron huecos solo durante el ajuste y se restauraron como invalidos despues.",
        "velocity_video": str(velocity_path),
        "vorticity_video": str(vorticity_path),
        "parameters": {
            "window_rows_x_cols": f"{args.window}x{args.window}",
            "window_length_each_axis": int(args.window),
            "polynomial_order": int(args.polyorder),
            "filter_mode": "separable, axis y then axis x, mode=interp",
            "base_min_correlation": min_corr,
            "temporal_filter_radius": radius,
            "temporal_filter_measurements": temporal_measurements,
            "vorticity_gaussian_sigma_grid_nodes": float(args.vorticity_sigma),
            "vorticity_threshold": None,
        },
        "input_valid_fraction": float(np.mean(valid)),
        "speed_color_limit_global_max_mm_s": speed_limit,
        "vorticity_color_limit_global_max_abs_1_s": omega_global_max,
        "vorticity_color_limit_mode": (
            "fixed_symmetric" if args.vorticity_color_limit is not None else "global_absolute_maximum"
        ),
        "vorticity_color_limit_abs_1_s": omega_limit,
        "passes": summary["passes"],
        "final_grid_step_px": final_step_px,
        "output_fps": output_fps,
        "output_duration_s": render_frames / output_fps,
        "plate_mask_polygon_px": plate_polygon.tolist() if plate_polygon is not None else None,
    }
    (args.out_dir / args.report_file).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
