import argparse
import csv
import json
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2

# Keep matplotlib cache inside the workspace on restricted Windows setups.
os.environ.setdefault("MPLCONFIGDIR", str(Path("resultados") / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    from scipy.fft import fftshift, irfft2, next_fast_len, rfft2
except ImportError:  # La ruta directa no necesita SciPy.
    fftshift = irfft2 = next_fast_len = rfft2 = None

TURBO_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0, :]
FFT_WINDOWS = {
    window: np.outer(np.hanning(window), np.hanning(window)).astype(np.float32)
    for window in (16, 24, 32, 48, 64, 96)
}
FFT_WORKERS = 1
FFT_BATCH_SIZE = 1024
DEFAULT_PASSES = [
    {"window": 64, "overlap": 0.5, "search": 8},
    {"window": 32, "overlap": 0.5, "search": 4},
]


@dataclass
class PassConfig:
    window: int
    overlap: float
    search: int


def load_config(path: Path) -> dict:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {frame_index}.")
    return frame


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=9, sigmaY=9)
    high_pass = cv2.addWeighted(enhanced, 1.4, background, -0.4, 0)
    lo, hi = np.percentile(high_pass, [1, 99])
    if hi <= lo:
        return high_pass.astype(np.float32) / 255.0
    norm = np.clip((high_pass.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return norm.astype(np.float32)


class PreprocessedFrameCache:
    def __init__(self, cap: cv2.VideoCapture, max_size: int = 24):
        self.cap = cap
        self.max_size = max(2, int(max_size))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_index: int) -> np.ndarray:
        if frame_index in self.cache:
            self.cache.move_to_end(frame_index)
            return self.cache[frame_index]
        frame = read_frame(self.cap, frame_index)
        preprocessed = preprocess_frame(frame)
        self.cache[frame_index] = preprocessed
        self.cache.move_to_end(frame_index)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return preprocessed


def parse_rect(rect) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    return int(x), int(y), int(w), int(h)


def build_mask(
    shape: tuple[int, int],
    rects: list[list[int]],
    polygons: list[list[list[int]]] | None = None,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for rect in rects or []:
        x, y, rw, rh = parse_rect(rect)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    for polygon in polygons or []:
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("Cada mask_polygon debe contener al menos tres puntos [x, y].")
        cv2.fillPoly(mask, [points], 1)
    return mask


def auto_calibrate_plate(
    frame: np.ndarray,
    plate_roi: list[int],
    plate_length_mm: float,
    out_path: Path,
) -> dict:
    x, y, w, h = parse_rect(plate_roi)
    roi = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 50, 40), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []
    for idx in range(1, num):
        cx, cy = centroids[idx]
        bx, by, bw, bh, area = stats[idx]
        if area >= 20:
            candidates.append(
                {
                    "area": int(area),
                    "bbox": [int(x + bx), int(y + by), int(bw), int(bh)],
                    "centroid": [float(x + cx), float(y + cy)],
                }
            )
    if not candidates:
        raise RuntimeError("No se detectó la placa en plate_roi.")

    best = max(candidates, key=lambda c: c["area"])
    bx, by, bw, bh = best["bbox"]
    plate_px = float(max(bw, bh))
    mm_per_px = plate_length_mm / plate_px

    debug = frame.copy()
    cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    cv2.putText(
        debug,
        f"{plate_px:.1f}px = {plate_length_mm:g}mm",
        (bx, max(20, by - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), debug)

    return {
        "mode": "auto_plate_roi",
        "plate_length_mm": plate_length_mm,
        "plate_px": plate_px,
        "mm_per_px": mm_per_px,
        "best_component": best,
        "candidates": candidates,
    }


def manual_calibrate_plate(
    frame: np.ndarray,
    plate_points: list[list[float]],
    plate_length_mm: float,
    out_path: Path,
) -> dict:
    (x1, y1), (x2, y2) = plate_points
    plate_px = float(np.hypot(x2 - x1, y2 - y1))
    if plate_px <= 0:
        raise RuntimeError("plate_points no puede tener distancia cero.")
    mm_per_px = plate_length_mm / plate_px

    debug = frame.copy()
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.line(debug, p1, p2, (0, 0, 255), 2)
    cv2.circle(debug, p1, 5, (255, 255, 0), -1)
    cv2.circle(debug, p2, 5, (255, 255, 0), -1)
    cv2.putText(
        debug,
        f"{plate_px:.1f}px = {plate_length_mm:g}mm",
        (min(p1[0], p2[0]), max(20, min(p1[1], p2[1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), debug)

    return {
        "mode": "manual_two_points",
        "plate_length_mm": plate_length_mm,
        "plate_px": plate_px,
        "mm_per_px": mm_per_px,
        "plate_points": plate_points,
    }


def gaussian_subpixel(corr: np.ndarray, peak_x: int, peak_y: int) -> tuple[float, float]:
    def axis_offset(v_minus: float, v0: float, v_plus: float) -> float:
        eps = 1e-9
        if v_minus <= 0 or v0 <= 0 or v_plus <= 0:
            denom = v_minus - 2 * v0 + v_plus
            if abs(denom) < eps:
                return 0.0
            return float(np.clip(0.5 * (v_minus - v_plus) / denom, -1, 1))
        lm, l0, lp = np.log(v_minus + eps), np.log(v0 + eps), np.log(v_plus + eps)
        denom = 2 * (lm - 2 * l0 + lp)
        if abs(denom) < eps:
            return 0.0
        return float(np.clip((lm - lp) / denom, -1, 1))

    h, w = corr.shape
    off_x = 0.0
    off_y = 0.0
    if 0 < peak_x < w - 1:
        off_x = axis_offset(corr[peak_y, peak_x - 1], corr[peak_y, peak_x], corr[peak_y, peak_x + 1])
    if 0 < peak_y < h - 1:
        off_y = axis_offset(corr[peak_y - 1, peak_x], corr[peak_y, peak_x], corr[peak_y + 1, peak_x])
    return off_x, off_y


def make_grid(roi: list[int], window: int, overlap: float) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = parse_rect(roi)
    step = max(1, int(round(window * (1 - overlap))))
    xs = np.arange(x + window // 2, x + w - window // 2 + 1, step, dtype=np.float32)
    ys = np.arange(y + window // 2, y + h - window // 2 + 1, step, dtype=np.float32)
    return np.meshgrid(xs, ys)


def bilinear_predict(
    prev_x: np.ndarray | None,
    prev_y: np.ndarray | None,
    prev_u: np.ndarray | None,
    prev_v: np.ndarray | None,
    x: float,
    y: float,
) -> tuple[float, float]:
    if prev_x is None or prev_y is None or prev_u is None or prev_v is None:
        return 0.0, 0.0

    xs = prev_x[0, :]
    ys = prev_y[:, 0]
    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
        return 0.0, 0.0

    ix = int(np.searchsorted(xs, x) - 1)
    iy = int(np.searchsorted(ys, y) - 1)
    ix = int(np.clip(ix, 0, len(xs) - 2))
    iy = int(np.clip(iy, 0, len(ys) - 2))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = 0.0 if x1 == x0 else float((x - x0) / (x1 - x0))
    ty = 0.0 if y1 == y0 else float((y - y0) / (y1 - y0))

    def interp(field: np.ndarray) -> float:
        vals = np.array(
            [
                field[iy, ix],
                field[iy, ix + 1],
                field[iy + 1, ix],
                field[iy + 1, ix + 1],
            ],
            dtype=np.float32,
        )
        if np.any(~np.isfinite(vals)):
            return 0.0
        return float(
            (1 - tx) * (1 - ty) * vals[0]
            + tx * (1 - ty) * vals[1]
            + (1 - tx) * ty * vals[2]
            + tx * ty * vals[3]
        )

    return interp(prev_u), interp(prev_v)


def fill_invalid_predictor(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rellena huecos del campo anterior solo para construir el predictor."""
    valid = np.isfinite(u) & np.isfinite(v)
    if valid.all():
        return u, v
    if not valid.any():
        return np.zeros_like(u, dtype=np.float32), np.zeros_like(v, dtype=np.float32)

    invalid_mask = (~valid).astype(np.uint8)
    filled_u = cv2.inpaint(np.where(valid, u, 0.0).astype(np.float32), invalid_mask, 3, cv2.INPAINT_NS)
    filled_v = cv2.inpaint(np.where(valid, v, 0.0).astype(np.float32), invalid_mask, 3, cv2.INPAINT_NS)
    return filled_u.astype(np.float32), filled_v.astype(np.float32)


def _windows_at_step(
    image: np.ndarray,
    roi: list[int],
    window: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, w, h = parse_rect(roi)
    image_h, image_w = image.shape[:2]
    if x < 0 or y < 0 or x + w > image_w or y + h > image_h:
        raise RuntimeError("El ROI PIV queda fuera de los limites del frame.")
    if w < window or h < window:
        raise RuntimeError(f"El ROI debe ser al menos de {window}x{window} pixeles.")

    roi_view = image[y : y + h, x : x + w]
    windows = sliding_window_view(roi_view, (window, window))[::step, ::step]
    rows, cols = windows.shape[:2]
    centers_x = x + window // 2 + np.arange(cols, dtype=np.float32) * step
    centers_y = y + window // 2 + np.arange(rows, dtype=np.float32) * step
    x_grid, y_grid = np.meshgrid(centers_x, centers_y)
    return windows, x_grid, y_grid


def fft_padded_size(window: int, search: int) -> int:
    """Tamano rapido que conserva correlaciones lineales dentro de +/- search."""
    minimum_size = window + max(0, search)
    return int(next_fast_len(minimum_size)) if next_fast_len is not None else minimum_size


def fft_batch_phase_correlate(
    img_a: np.ndarray,
    img_b: np.ndarray,
    valid_mask: np.ndarray,
    roi: list[int],
    cfg: PassConfig,
    min_corr: float,
    prev: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correlacion FFT por lotes para una pasada de la malla PIV."""
    if rfft2 is None or irfft2 is None or fftshift is None:
        raise RuntimeError("El modo FFT requiere SciPy. Instala las dependencias de requisitos.txt.")
    window = int(cfg.window)
    if window not in FFT_WINDOWS:
        raise RuntimeError("El modo FFT por lotes admite ventanas de 16, 32, 48, 64 o 96 px.")
    step = max(1, int(round(window * (1.0 - cfg.overlap))))
    padded_size = fft_padded_size(window, cfg.search)
    hann_window = FFT_WINDOWS[window]

    windows_a, x_grid, y_grid = _windows_at_step(img_a, roi, window, step)

    rows, cols = windows_a.shape[:2]
    count = rows * cols
    u = np.full((rows, cols), np.nan, dtype=np.float32)
    v = np.full((rows, cols), np.nan, dtype=np.float32)
    corr_peak = np.full((rows, cols), np.nan, dtype=np.float32)
    if count == 0:
        return x_grid, y_grid, u, v, corr_peak

    windows_b = None
    if prev is None:
        windows_b, _, _ = _windows_at_step(img_b, roi, window, step)
        if windows_a.shape != windows_b.shape:
            raise RuntimeError("Los frames del par no generan la misma grilla de ventanas.")

    roi_x, roi_y, roi_w, roi_h = parse_rect(roi)
    valid_roi = valid_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    valid_windows = sliding_window_view(valid_roi, (window, window))[::step, ::step]
    base_good = valid_windows.mean(axis=(2, 3)).reshape(-1) >= 0.75
    search = int(np.clip(cfg.search, 0, padded_size // 2 - 1))
    center = padded_size // 2
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    corr_flat = corr_peak.reshape(-1)
    flat_x = x_grid.reshape(-1)
    flat_y = y_grid.reshape(-1)
    prev_x = prev_y = prev_u = prev_v = None
    if prev is not None:
        prev_x, prev_y, prev_u, prev_v = prev

    for batch_start in range(0, count, FFT_BATCH_SIZE):
        batch_stop = min(count, batch_start + FFT_BATCH_SIZE)
        flat_indices = np.arange(batch_start, batch_stop)
        row_indices, col_indices = np.divmod(flat_indices, cols)
        batch_count = batch_stop - batch_start

        batch_a = np.array(windows_a[row_indices, col_indices], dtype=np.float32, copy=True)
        batch_b = np.zeros((batch_count, window, window), dtype=np.float32)
        base_dx = np.zeros(batch_count, dtype=np.float32)
        base_dy = np.zeros(batch_count, dtype=np.float32)
        target_good = np.ones(batch_count, dtype=bool)

        if windows_b is not None:
            batch_b[:] = windows_b[row_indices, col_indices]
        else:
            half = window // 2
            img_h, img_w = img_b.shape[:2]
            for local_index, global_index in enumerate(flat_indices):
                cx = float(flat_x[global_index])
                cy = float(flat_y[global_index])
                x0 = int(round(cx - half))
                y0 = int(round(cy - half))
                pred_u, pred_v = bilinear_predict(prev_x, prev_y, prev_u, prev_v, cx, cy)
                bx0 = int(round(cx + pred_u - half))
                by0 = int(round(cy + pred_v - half))
                bx1 = bx0 + window
                by1 = by0 + window
                if bx0 < 0 or by0 < 0 or bx1 > img_w or by1 > img_h:
                    target_good[local_index] = False
                    continue
                batch_b[local_index] = img_b[by0:by1, bx0:bx1]
                base_dx[local_index] = float(bx0 - x0)
                base_dy[local_index] = float(by0 - y0)

        good = base_good[flat_indices] & target_good
        good &= np.std(batch_a, axis=(1, 2)) >= 0.01
        good &= np.std(batch_b, axis=(1, 2)) >= 0.01

        batch_a -= batch_a.mean(axis=(1, 2), keepdims=True)
        batch_b -= batch_b.mean(axis=(1, 2), keepdims=True)
        batch_a *= hann_window
        batch_b *= hann_window
        padded_a = np.zeros((batch_count, padded_size, padded_size), dtype=np.float32)
        padded_b = np.zeros_like(padded_a)
        padded_a[:, :window, :window] = batch_a
        padded_b[:, :window, :window] = batch_b

        spectrum_a = rfft2(padded_a, axes=(-2, -1), workers=FFT_WORKERS)
        spectrum_b = rfft2(padded_b, axes=(-2, -1), workers=FFT_WORKERS)
        correlation = irfft2(
            np.conj(spectrum_a) * spectrum_b,
            s=(padded_size, padded_size),
            axes=(-2, -1),
            workers=FFT_WORKERS,
        )
        correlation = fftshift(correlation, axes=(-2, -1))
        energy = np.sqrt(
            np.sum(batch_a * batch_a, axis=(1, 2)) * np.sum(batch_b * batch_b, axis=(1, 2))
        )
        correlation /= np.maximum(energy[:, None, None], np.finfo(np.float32).eps)

        search_slice = correlation[
            :, center - search : center + search + 1, center - search : center + search + 1
        ]
        peak_flat = np.argmax(search_slice.reshape(batch_count, -1), axis=1)
        peak_y_local, peak_x_local = np.divmod(peak_flat, 2 * search + 1)
        peak_x = peak_x_local + center - search
        peak_y = peak_y_local + center - search
        max_values = search_slice.reshape(batch_count, -1)[np.arange(batch_count), peak_flat]
        good &= np.isfinite(max_values) & (max_values >= min_corr)

        for local_index in np.flatnonzero(good):
            offset_x, offset_y = gaussian_subpixel(
                correlation[local_index],
                int(peak_x[local_index]),
                int(peak_y[local_index]),
            )
            dx = float(peak_x[local_index] - center) + offset_x
            dy = float(peak_y[local_index] - center) + offset_y
            if abs(dx) <= cfg.search and abs(dy) <= cfg.search:
                global_index = flat_indices[local_index]
                u_flat[global_index] = base_dx[local_index] + dx
                v_flat[global_index] = base_dy[local_index] + dy
                corr_flat[global_index] = float(max_values[local_index])

    return x_grid, y_grid, u, v, corr_peak


def run_pass(
    img_a: np.ndarray,
    img_b: np.ndarray,
    valid_mask: np.ndarray,
    roi: list[int],
    cfg: PassConfig,
    prev: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    min_corr: float,
    correlation_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if correlation_method == "fft":
        return fft_batch_phase_correlate(img_a, img_b, valid_mask, roi, cfg, min_corr, prev)

    x_grid, y_grid = make_grid(roi, cfg.window, cfg.overlap)
    u = np.full_like(x_grid, np.nan, dtype=np.float32)
    v = np.full_like(y_grid, np.nan, dtype=np.float32)
    corr_peak = np.full_like(x_grid, np.nan, dtype=np.float32)

    prev_x = prev_y = prev_u = prev_v = None
    if prev is not None:
        prev_x, prev_y, prev_u, prev_v = prev

    half = cfg.window // 2
    h, w = img_a.shape
    for row in range(x_grid.shape[0]):
        for col in range(x_grid.shape[1]):
            cx = float(x_grid[row, col])
            cy = float(y_grid[row, col])
            x0 = int(round(cx - half))
            y0 = int(round(cy - half))
            x1 = x0 + cfg.window
            y1 = y0 + cfg.window
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            if valid_mask[y0:y1, x0:x1].mean() < 0.75:
                continue

            pred_u, pred_v = bilinear_predict(prev_x, prev_y, prev_u, prev_v, cx, cy)
            templ = img_a[y0:y1, x0:x1]
            if float(np.std(templ)) < 0.01:
                continue
            sx0 = int(round(cx + pred_u - half - cfg.search))
            sy0 = int(round(cy + pred_v - half - cfg.search))
            sx1 = sx0 + cfg.window + 2 * cfg.search
            sy1 = sy0 + cfg.window + 2 * cfg.search
            if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
                continue
            search = img_b[sy0:sy1, sx0:sx1]
            if float(np.std(search)) < 0.01:
                continue
            corr = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(corr)
            px, py = max_loc
            off_x, off_y = gaussian_subpixel(corr, px, py)
            dx = sx0 + px + off_x - x0
            dy = sy0 + py + off_y - y0
            if max_val >= min_corr:
                u[row, col] = dx
                v[row, col] = dy
                corr_peak[row, col] = max_val

    return x_grid, y_grid, u, v, corr_peak


def local_median_validate(
    u: np.ndarray,
    v: np.ndarray,
    corr: np.ndarray,
    min_corr: float,
    threshold_px: float,
) -> np.ndarray:
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(corr) & (corr >= min_corr)
    out = valid.copy()
    rows, cols = u.shape
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            r0, r1 = max(0, r - 1), min(rows, r + 2)
            c0, c1 = max(0, c - 1), min(cols, c + 2)
            nu = u[r0:r1, c0:c1]
            nv = v[r0:r1, c0:c1]
            nvalid = valid[r0:r1, c0:c1]
            if int(nvalid.sum()) < 4:
                continue
            med_u = float(np.nanmedian(np.where(nvalid, nu, np.nan)))
            med_v = float(np.nanmedian(np.where(nvalid, nv, np.nan)))
            if np.hypot(float(u[r, c] - med_u), float(v[r, c] - med_v)) > threshold_px:
                out[r, c] = False
    return out


def save_vectors_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_pair_count(pairs_config, start_frame: int, frame_delta: int, stride: int, frame_count: int) -> int:
    if isinstance(pairs_config, str) and pairs_config.lower() in {"all", "todo", "*"}:
        last_start = frame_count - 1 - frame_delta
        available = last_start - start_frame
        if available < 0:
            return 0
        return available // stride + 1
    return int(pairs_config)


def draw_overlay(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    title: str,
    out_path: Path,
    display_scale: float = 6.0,
) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(7.2, 12.8), dpi=120)
    ax.imshow(rgb)
    ax.set_title(title)
    ax.set_axis_off()
    speed = np.hypot(u, v)
    q = ax.quiver(
        x[valid],
        y[valid],
        u[valid] * display_scale,
        v[valid] * display_scale,
        speed[valid],
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="turbo",
        width=0.003,
    )
    fig.colorbar(q, ax=ax, fraction=0.035, pad=0.01, label="desplazamiento real [px/frame]")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def color_from_turbo(value: float, vmax: float) -> tuple[int, int, int]:
    if not np.isfinite(value) or vmax <= 0:
        idx = 0
    else:
        idx = int(np.clip(round(255 * value / vmax), 0, 255))
    color = TURBO_LUT[idx]
    return int(color[0]), int(color[1]), int(color[2])


def draw_colorbar(frame: np.ndarray, vmax: float, label: str) -> None:
    h, w = frame.shape[:2]
    x0 = max(8, w - 54)
    y0 = 92
    bar_w = 16
    bar_h = min(260, max(120, h - 210))
    for i in range(bar_h):
        frac = 1.0 - i / max(1, bar_h - 1)
        color = color_from_turbo(frac * vmax, vmax)
        cv2.line(frame, (x0, y0 + i), (x0 + bar_w, y0 + i), color, 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (230, 230, 230), 1)
    cv2.putText(frame, f"{vmax:g}", (x0 - 4, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.putText(frame, "0", (x0 + 2, y0 + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.putText(frame, label, (x0 - 90, y0 + bar_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)


def draw_vector_video_frame(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    roi: list[int],
    mask_rects: list[list[int]],
    frame_a: int,
    frame_b: int,
    time_s: float,
    settings: dict,
) -> np.ndarray:
    out = frame.copy()
    display_scale = float(settings.get("display_scale", 6.0))
    vector_step = max(1, int(settings.get("vector_step", 1)))
    color_max = float(settings.get("color_max_px_frame", 8.0))
    thickness = max(1, int(settings.get("thickness", 1)))
    fixed_arrow_color = settings.get("arrow_color_bgr")
    if fixed_arrow_color is not None:
        if not isinstance(fixed_arrow_color, (list, tuple)) or len(fixed_arrow_color) != 3:
            raise ValueError("arrow_color_bgr debe contener tres valores BGR")
        fixed_arrow_color = tuple(int(np.clip(value, 0, 255)) for value in fixed_arrow_color)

    if settings.get("draw_roi", True):
        rx, ry, rw, rh = parse_rect(roi)
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)
    if settings.get("draw_masks", True):
        for rect in mask_rects:
            mx, my, mw, mh = parse_rect(rect)
            cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (255, 0, 255), 1)

    rows, cols = x.shape
    drawn = 0
    valid_count = int(valid.sum())
    for row in range(0, rows, vector_step):
        for col in range(0, cols, vector_step):
            if not valid[row, col]:
                continue
            dx = float(u[row, col])
            dy = float(v[row, col])
            if not np.isfinite(dx) or not np.isfinite(dy):
                continue
            speed = float(np.hypot(dx, dy))
            start = (int(round(float(x[row, col]))), int(round(float(y[row, col]))))
            end = (
                int(round(float(x[row, col]) + dx * display_scale)),
                int(round(float(y[row, col]) + dy * display_scale)),
            )
            cv2.arrowedLine(
                out,
                start,
                end,
                fixed_arrow_color if fixed_arrow_color is not None else color_from_turbo(speed, color_max),
                thickness,
                cv2.LINE_AA,
                tipLength=0.28,
            )
            drawn += 1

    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (430, 76), (0, 0, 0), -1)
    out = cv2.addWeighted(panel, 0.45, out, 0.55, 0)
    cv2.putText(
        out,
        f"PIV frame {frame_a}->{frame_b}  t={time_s:.3f}s",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"valid={valid_count}  drawn={drawn}  arrow x{display_scale:g}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if fixed_arrow_color is None:
        draw_colorbar(out, color_max, "px/frame")
    else:
        label = str(settings.get("arrow_color_label", "color fijo"))
        cv2.putText(
            out,
            f"vectores: {label}",
            (max(8, out.shape[1] - 190), 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            fixed_arrow_color,
            1,
            cv2.LINE_AA,
        )
    return out


def open_vector_video_writer(
    out_dir: Path,
    width: int,
    height: int,
    fps: float,
    stride: int,
    settings: dict,
) -> tuple[cv2.VideoWriter | None, Path | None, float | None]:
    if not settings.get("enabled", False):
        return None, None, None
    file_name = settings.get("file", "vectors_overlay.mp4")
    path = out_dir / file_name
    video_fps = settings.get("fps")
    if video_fps is None:
        video_fps = fps / max(1, stride)
    video_fps = float(video_fps)
    fourcc = cv2.VideoWriter_fourcc(*settings.get("fourcc", "mp4v"))
    writer = cv2.VideoWriter(str(path), fourcc, video_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de vectores: {path}")
    return writer, path, video_fps


def process_pair_preprocessed(
    pre_a: np.ndarray,
    pre_b: np.ndarray,
    valid_mask: np.ndarray,
    roi: list[int],
    passes: list[PassConfig],
    min_corr: float,
    local_threshold_px: float,
    correlation_method: str,
    spatial_filter_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev = None
    x = y = u = v = corr = None
    for pass_index, cfg in enumerate(passes):
        x, y, u, v, corr = run_pass(pre_a, pre_b, valid_mask, roi, cfg, prev, min_corr, correlation_method)
        if spatial_filter_enabled:
            valid = local_median_validate(u, v, corr, min_corr, local_threshold_px)
        else:
            valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(corr) & (corr >= min_corr)
        u = np.where(valid, u, np.nan).astype(np.float32)
        v = np.where(valid, v, np.nan).astype(np.float32)
        if pass_index < len(passes) - 1:
            predictor_u, predictor_v = fill_invalid_predictor(u, v)
            prev = (x, y, predictor_u, predictor_v)

    if spatial_filter_enabled:
        valid = local_median_validate(u, v, corr, min_corr, local_threshold_px)
    else:
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(corr) & (corr >= min_corr)
    return x, y, u, v, corr, valid


def process_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    roi: list[int],
    mask_rects: list[list[int]],
    passes: list[PassConfig],
    min_corr: float,
    local_threshold_px: float,
    correlation_method: str = "fft",
    mask_polygons: list[list[list[int]]] | None = None,
    spatial_filter_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre_a = preprocess_frame(frame_a)
    pre_b = preprocess_frame(frame_b)
    solid_mask = build_mask(pre_a.shape, mask_rects, mask_polygons)
    valid_mask = (solid_mask == 0).astype(np.uint8)
    return process_pair_preprocessed(
        pre_a,
        pre_b,
        valid_mask,
        roi,
        passes,
        min_corr,
        local_threshold_px,
        correlation_method,
        spatial_filter_enabled,
    )


def apply_temporal_filter(pair_results: list[dict], settings: dict) -> list[dict]:
    if not settings.get("enabled", False) or not pair_results:
        for result in pair_results:
            result["u_out"] = result["u"]
            result["v_out"] = result["v"]
            result["valid_out"] = result["valid"]
        return pair_results

    radius = max(0, int(settings.get("radius", 2)))
    method = str(settings.get("method", "median")).lower()
    min_count = max(1, int(settings.get("min_count", 3)))

    u_stack = np.stack([np.where(r["valid"], r["u"], np.nan) for r in pair_results], axis=0)
    v_stack = np.stack([np.where(r["valid"], r["v"], np.nan) for r in pair_results], axis=0)
    finite_stack = np.isfinite(u_stack) & np.isfinite(v_stack)

    for idx, result in enumerate(pair_results):
        lo = max(0, idx - radius)
        hi = min(len(pair_results), idx + radius + 1)
        window_u = u_stack[lo:hi]
        window_v = v_stack[lo:hi]
        window_valid = finite_stack[lo:hi]
        required = min(min_count, hi - lo)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if method == "mean":
                u_out = np.nanmean(window_u, axis=0)
                v_out = np.nanmean(window_v, axis=0)
            else:
                u_out = np.nanmedian(window_u, axis=0)
                v_out = np.nanmedian(window_v, axis=0)
        count = np.sum(window_valid, axis=0)
        valid_out = (count >= required) & np.isfinite(u_out) & np.isfinite(v_out)
        result["u_out"] = np.where(valid_out, u_out, np.nan).astype(np.float32)
        result["v_out"] = np.where(valid_out, v_out, np.nan).astype(np.float32)
        result["valid_out"] = valid_out

    return pair_results


_WORKER_FRAME_CACHE: PreprocessedFrameCache | None = None
_WORKER_ROI: list[int] | None = None
_WORKER_MASK_RECTS: list[list[int]] | None = None
_WORKER_MASK_POLYGONS: list[list[list[int]]] | None = None
_WORKER_PASSES: list[PassConfig] | None = None
_WORKER_MIN_CORR: float | None = None
_WORKER_LOCAL_THRESHOLD_PX: float | None = None
_WORKER_VALID_MASK: np.ndarray | None = None
_WORKER_CORRELATION_METHOD: str | None = None
_WORKER_SPATIAL_FILTER_ENABLED: bool = True


def init_piv_worker(
    video_path: str,
    roi: list[int],
    mask_rects: list[list[int]],
    mask_polygons: list[list[list[int]]],
    passes_payload: list[dict],
    min_corr: float,
    local_threshold_px: float,
    correlation_method: str,
    frame_cache_size: int,
    spatial_filter_enabled: bool,
) -> None:
    global _WORKER_FRAME_CACHE
    global _WORKER_ROI
    global _WORKER_MASK_RECTS
    global _WORKER_MASK_POLYGONS
    global _WORKER_PASSES
    global _WORKER_MIN_CORR
    global _WORKER_LOCAL_THRESHOLD_PX
    global _WORKER_VALID_MASK
    global _WORKER_CORRELATION_METHOD
    global _WORKER_SPATIAL_FILTER_ENABLED

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video en worker: {video_path}")
    _WORKER_FRAME_CACHE = PreprocessedFrameCache(cap, frame_cache_size)
    _WORKER_ROI = roi
    _WORKER_MASK_RECTS = mask_rects
    _WORKER_MASK_POLYGONS = mask_polygons
    _WORKER_PASSES = [PassConfig(**item) for item in passes_payload]
    _WORKER_MIN_CORR = float(min_corr)
    _WORKER_LOCAL_THRESHOLD_PX = float(local_threshold_px)
    _WORKER_CORRELATION_METHOD = correlation_method
    _WORKER_SPATIAL_FILTER_ENABLED = bool(spatial_filter_enabled)
    _WORKER_VALID_MASK = None


def worker_valid_mask(shape: tuple[int, int]) -> np.ndarray:
    global _WORKER_VALID_MASK
    if _WORKER_MASK_RECTS is None:
        raise RuntimeError("Worker PIV no inicializado.")
    if _WORKER_VALID_MASK is None or _WORKER_VALID_MASK.shape != shape:
        solid_mask = build_mask(shape, _WORKER_MASK_RECTS, _WORKER_MASK_POLYGONS)
        _WORKER_VALID_MASK = (solid_mask == 0).astype(np.uint8)
    return _WORKER_VALID_MASK


def process_pair_task(task: tuple[int, int, int, float]) -> dict:
    if (
        _WORKER_FRAME_CACHE is None
        or _WORKER_ROI is None
        or _WORKER_PASSES is None
        or _WORKER_MIN_CORR is None
        or _WORKER_LOCAL_THRESHOLD_PX is None
        or _WORKER_CORRELATION_METHOD is None
    ):
        raise RuntimeError("Worker PIV no inicializado.")

    pair_idx, fa, fb, fps = task
    pre_a = _WORKER_FRAME_CACHE.get(fa)
    pre_b = _WORKER_FRAME_CACHE.get(fb)
    valid_mask = worker_valid_mask(pre_a.shape)
    x, y, u, v, corr, valid = process_pair_preprocessed(
        pre_a,
        pre_b,
        valid_mask,
        _WORKER_ROI,
        _WORKER_PASSES,
        _WORKER_MIN_CORR,
        _WORKER_LOCAL_THRESHOLD_PX,
        _WORKER_CORRELATION_METHOD,
        _WORKER_SPATIAL_FILTER_ENABLED,
    )
    return {
        "pair_idx": pair_idx,
        "frame_a": fa,
        "frame_b": fb,
        "time_s": fa / fps if fps > 0 else float("nan"),
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "corr": corr,
        "valid": valid,
    }


def resolve_parallel_settings(settings: dict, pair_count: int) -> tuple[bool, int, int, int]:
    if isinstance(settings, bool):
        settings = {"enabled": settings}
    enabled = bool(settings.get("enabled", True))
    if not enabled or pair_count <= 1:
        return False, 1, 1, int(settings.get("frame_cache_size", 24))

    cpu_count = os.cpu_count() or 1
    auto_workers = max(1, min(pair_count, max(1, cpu_count - 1), int(settings.get("auto_max_workers", 4))))
    workers_value = settings.get("workers", "auto")
    if isinstance(workers_value, str) and workers_value.lower() in {"auto", ""}:
        workers = auto_workers
    else:
        workers = int(workers_value)
    workers = max(1, min(workers, pair_count))

    chunksize = max(1, int(settings.get("chunksize", 8)))
    frame_cache_size = max(2, int(settings.get("frame_cache_size", max(24, chunksize + 2))))
    return workers > 1, workers, chunksize, frame_cache_size


def build_pair_tasks(
    start_frame: int,
    frame_delta: int,
    stride: int,
    frame_count: int,
    pairs: int,
    fps: float,
) -> list[tuple[int, int, int, float]]:
    tasks = []
    for pair_idx in range(pairs):
        fa = start_frame + pair_idx * stride
        fb = fa + frame_delta
        if fb >= frame_count:
            break
        tasks.append((pair_idx, fa, fb, fps))
    return tasks


def compute_pair_results_serial(
    cap: cv2.VideoCapture,
    tasks: list[tuple[int, int, int, float]],
    roi: list[int],
    mask_rects: list[list[int]],
    mask_polygons: list[list[list[int]]],
    passes: list[PassConfig],
    min_corr: float,
    local_threshold_px: float,
    correlation_method: str,
    frame_cache_size: int,
    spatial_filter_enabled: bool,
) -> list[dict]:
    frame_cache = PreprocessedFrameCache(cap, frame_cache_size)
    valid_mask = None
    pair_results = []
    total = len(tasks)
    for done, (pair_idx, fa, fb, fps) in enumerate(tasks, 1):
        print(f"Procesando par {done}/{total}: frame {fa} -> {fb}", file=sys.stderr)
        pre_a = frame_cache.get(fa)
        pre_b = frame_cache.get(fb)
        if valid_mask is None or valid_mask.shape != pre_a.shape:
            solid_mask = build_mask(pre_a.shape, mask_rects, mask_polygons)
            valid_mask = (solid_mask == 0).astype(np.uint8)
        x, y, u, v, corr, valid = process_pair_preprocessed(
            pre_a,
            pre_b,
            valid_mask,
            roi,
            passes,
            min_corr,
            local_threshold_px,
            correlation_method,
            spatial_filter_enabled,
        )
        pair_results.append(
            {
                "pair_idx": pair_idx,
                "frame_a": fa,
                "frame_b": fb,
                "time_s": fa / fps if fps > 0 else float("nan"),
                "x": x,
                "y": y,
                "u": u,
                "v": v,
                "corr": corr,
                "valid": valid,
            }
        )
    return pair_results


def compute_pair_results_parallel(
    video_path: Path,
    tasks: list[tuple[int, int, int, float]],
    roi: list[int],
    mask_rects: list[list[int]],
    mask_polygons: list[list[list[int]]],
    passes: list[PassConfig],
    min_corr: float,
    local_threshold_px: float,
    correlation_method: str,
    workers: int,
    chunksize: int,
    frame_cache_size: int,
    checkpoint_dir: Path | None = None,
    spatial_filter_enabled: bool = True,
) -> list[dict]:
    print(
        f"Procesando {len(tasks)} pares PIV con {workers} procesos "
        f"(chunksize={chunksize}, cache={frame_cache_size} frames/worker)",
        file=sys.stderr,
    )
    passes_payload = [cfg.__dict__ for cfg in passes]
    results_by_pair: dict[int, dict] = {}
    pending_tasks = tasks
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        pending_tasks = []
        for task in tasks:
            pair_idx = task[0]
            checkpoint_path = checkpoint_dir / f"pair_{pair_idx:06d}.pkl"
            if checkpoint_path.exists():
                with checkpoint_path.open("rb") as fh:
                    results_by_pair[pair_idx] = pickle.load(fh)
            else:
                pending_tasks.append(task)
        if results_by_pair:
            print(
                f"Reanudando desde {len(results_by_pair)} checkpoints; "
                f"faltan {len(pending_tasks)} pares.",
                file=sys.stderr,
            )

    if not pending_tasks:
        return [results_by_pair[task[0]] for task in tasks]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_piv_worker,
        initargs=(
            str(video_path.resolve()),
            roi,
            mask_rects,
            mask_polygons,
            passes_payload,
            min_corr,
            local_threshold_px,
            correlation_method,
            frame_cache_size,
            spatial_filter_enabled,
        ),
    ) as executor:
        iterator = executor.map(process_pair_task, pending_tasks, chunksize=chunksize)
        for done, result in enumerate(iterator, 1):
            print(
                f"Par PIV listo {done}/{len(pending_tasks)}: "
                f"frame {result['frame_a']} -> {result['frame_b']}",
                file=sys.stderr,
            )
            pair_idx = result["pair_idx"]
            results_by_pair[pair_idx] = result
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / f"pair_{pair_idx:06d}.pkl"
                temp_path = checkpoint_path.with_suffix(".pkl.tmp")
                with temp_path.open("wb") as fh:
                    pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
                temp_path.replace(checkpoint_path)
    return [results_by_pair[task[0]] for task in tasks]


def main() -> int:
    parser = argparse.ArgumentParser(description="Procesador PIV multipasada para videos.")
    parser.add_argument("--config", type=Path, default=None, help="Archivo JSON de configuración.")
    parser.add_argument("--video", type=Path, default=None, help="Video MP4 de entrada.")
    parser.add_argument("--out", type=Path, default=None, help="Carpeta de salida.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    video_path = args.video or Path(config.get("video", "ensayo_placa_2.mp4"))
    out_dir = args.out or Path("resultados") / video_path.stem
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    metadata_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps_override = float(config.get("fps_override", 0.0) or 0.0)
    fps = fps_override if fps_override > 0 else metadata_fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(config.get("start_frame", 0))
    calibration_frame = int(config.get("calibration_frame", start_frame))
    frame_delta = int(config.get("frame_delta", 1))
    pairs_config = config.get("pairs", 1)
    stride_requested = int(config.get("stride", 2))
    if stride_requested < 0:
        raise RuntimeError("stride debe ser mayor o igual que cero.")
    stride = 1 if stride_requested == 0 else stride_requested
    pairs = resolve_pair_count(pairs_config, start_frame, frame_delta, stride, frame_count)
    overlay_every = int(config.get("overlay_every", 1))
    vector_video_settings = config.get("vector_video", {})
    if isinstance(vector_video_settings, bool):
        vector_video_settings = {"enabled": vector_video_settings}
    temporal_filter_settings = config.get("temporal_filter", {})
    if isinstance(temporal_filter_settings, bool):
        temporal_filter_settings = {"enabled": temporal_filter_settings}
    parallel_settings = config.get("parallel", {})
    if isinstance(parallel_settings, bool):
        parallel_settings = {"enabled": parallel_settings}
    roi = config.get("roi", [0, 0, width, height])
    mask_rects = config.get("mask_rects", [])
    mask_polygons = config.get("mask_polygons", [])
    plate_length_mm = float(config.get("plate_length_mm", 40.0))
    plate_roi = config.get("plate_roi")
    plate_points = config.get("plate_points")
    min_corr = float(config.get("min_corr", 0.18))
    local_threshold_px = float(config.get("local_threshold_px", 8.0))
    spatial_filter_enabled = bool(config.get("spatial_filter_enabled", True))
    correlation_method = str(config.get("correlation_method", "fft")).lower()
    if correlation_method not in {"direct", "fft"}:
        raise RuntimeError("correlation_method debe ser 'direct' o 'fft'.")
    passes = [PassConfig(**item) for item in config.get("passes", DEFAULT_PASSES)]

    if correlation_method == "fft":
        if rfft2 is None or irfft2 is None or fftshift is None:
            raise RuntimeError("El modo FFT requiere scipy. Instala requisitos.txt antes de continuar.")
        if len(passes) not in {1, 2}:
            raise RuntimeError("Para correlation_method='fft' usa una o dos pasadas.")
        for pass_config in passes:
            if pass_config.window not in FFT_WINDOWS:
                raise RuntimeError("FFT admite ventanas de 16, 24, 32, 48, 64 o 96 px.")
            if pass_config.window * (1.0 - pass_config.overlap) < 1.0:
                raise RuntimeError("El paso FFT debe ser al menos 1 pixel.")

    if calibration_frame < 0 or calibration_frame >= frame_count:
        raise RuntimeError("calibration_frame queda fuera del video.")
    ref_frame = read_frame(cap, calibration_frame)
    explicit_mm_per_px = config.get("mm_per_px")
    if explicit_mm_per_px is not None:
        mm_per_px_value = float(explicit_mm_per_px)
        if not np.isfinite(mm_per_px_value) or mm_per_px_value <= 0:
            raise RuntimeError("mm_per_px debe ser un valor positivo.")
        calibration = {
            "mode": "explicit_scale",
            "plate_length_mm": None,
            "plate_px": None,
            "mm_per_px": mm_per_px_value,
        }
    elif plate_points:
        calibration = manual_calibrate_plate(
            ref_frame,
            plate_points,
            plate_length_mm,
            out_dir / "calibration_debug.png",
        )
    elif plate_roi:
        calibration = auto_calibrate_plate(
            ref_frame,
            plate_roi,
            plate_length_mm,
            out_dir / "calibration_debug.png",
        )
    else:
        calibration = {
            "mode": "uncalibrated",
            "plate_length_mm": plate_length_mm,
            "plate_px": None,
            "mm_per_px": None,
        }

    mm_per_px = calibration.get("mm_per_px")
    dt = frame_delta / fps if fps > 0 else None
    fieldnames_px = [
        "frame_a",
        "frame_b",
        "time_s",
        "x_px",
        "y_px",
        "u_px_frame",
        "v_px_frame",
        "speed_px_frame",
        "corr",
        "valid",
    ]
    fieldnames_mm = fieldnames_px + ["x_mm", "y_mm", "u_mm_s", "v_mm_s", "speed_mm_s"]
    vectors_total = 0
    valid_count = 0
    tasks = build_pair_tasks(start_frame, frame_delta, stride, frame_count, pairs, fps)
    pairs_done = len(tasks)
    parallel_enabled, workers, chunksize, frame_cache_size = resolve_parallel_settings(parallel_settings, pairs_done)
    cv2.setNumThreads(1)
    if parallel_enabled:
        pair_results = compute_pair_results_parallel(
            video_path,
            tasks,
            roi,
            mask_rects,
            mask_polygons,
            passes,
            min_corr,
            local_threshold_px,
            correlation_method,
            workers,
            chunksize,
            frame_cache_size,
            out_dir / "pair_checkpoints",
            spatial_filter_enabled,
        )
    else:
        print(f"Procesando {pairs_done} pares PIV en modo serial con cache={frame_cache_size} frames", file=sys.stderr)
        pair_results = compute_pair_results_serial(
            cap,
            tasks,
            roi,
            mask_rects,
            mask_polygons,
            passes,
            min_corr,
            local_threshold_px,
            correlation_method,
            frame_cache_size,
            spatial_filter_enabled,
        )

    if temporal_filter_settings.get("enabled", False):
        print(
            "Aplicando filtro temporal "
            f"{temporal_filter_settings.get('method', 'median')} "
            f"radio={temporal_filter_settings.get('radius', 2)}",
            file=sys.stderr,
        )
    pair_results = apply_temporal_filter(pair_results, temporal_filter_settings)

    vector_video_path = None
    vector_video_fps = None
    write_px_csv = bool(config.get("write_px_csv", True))
    px_context = (
        (out_dir / "vectors_px.csv").open("w", newline="", encoding="utf-8")
        if write_px_csv
        else nullcontext(None)
    )
    with px_context as px_fh:
        px_writer = csv.DictWriter(px_fh, fieldnames=fieldnames_px) if px_fh is not None else None
        if px_writer is not None:
            px_writer.writeheader()
        mm_fh = None
        mm_writer = None
        if mm_per_px and dt:
            mm_fh = (out_dir / "vectors_mm_s.csv").open("w", newline="", encoding="utf-8")
            mm_writer = csv.DictWriter(mm_fh, fieldnames=fieldnames_mm)
            mm_writer.writeheader()

        try:
            for result in pair_results:
                pair_idx = result["pair_idx"]
                fa = result["frame_a"]
                fb = result["frame_b"]
                time_s = result["time_s"]
                x = result["x"]
                y = result["y"]
                u = result["u_out"]
                v = result["v_out"]
                corr = result["corr"]
                valid = result["valid_out"]

                if overlay_every > 0 and pair_idx % overlay_every == 0:
                    frame_a = video_uint8[fa] if video_uint8 is not None else read_frame(cap, fa)
                    overlay_name = f"overlay_f{fa:05d}_to_f{fb:05d}.png"
                    draw_overlay(
                        frame_a,
                        x,
                        y,
                        u,
                        v,
                        valid,
                        f"PIV frame {fa} -> {fb}",
                        out_dir / overlay_name,
                    )

                for idx in np.ndindex(x.shape):
                    is_valid = bool(valid[idx])
                    u_px = float(u[idx]) if np.isfinite(u[idx]) else float("nan")
                    v_px = float(v[idx]) if np.isfinite(v[idx]) else float("nan")
                    speed_px = float(np.hypot(u_px, v_px)) if is_valid else float("nan")
                    base = {
                        "frame_a": fa,
                        "frame_b": fb,
                        "time_s": time_s,
                        "x_px": float(x[idx]),
                        "y_px": float(y[idx]),
                        "u_px_frame": u_px,
                        "v_px_frame": v_px,
                        "speed_px_frame": speed_px,
                        "corr": float(corr[idx]) if np.isfinite(corr[idx]) else float("nan"),
                        "valid": int(is_valid),
                    }
                    if px_writer is not None:
                        px_writer.writerow(base)
                    if mm_writer is not None:
                        mm_writer.writerow(
                            {
                                **base,
                                "x_mm": float(x[idx] * mm_per_px),
                                "y_mm": float(y[idx] * mm_per_px),
                                "u_mm_s": float(u_px * mm_per_px / dt) if is_valid else float("nan"),
                                "v_mm_s": float(v_px * mm_per_px / dt) if is_valid else float("nan"),
                                "speed_mm_s": float(speed_px * mm_per_px / dt) if is_valid else float("nan"),
                            }
                        )
                    vectors_total += 1
                    valid_count += int(is_valid)
        finally:
            if mm_fh is not None:
                mm_fh.close()

    # El video se reconstruye al final, leyendo los frames originales uint8 y usando los campos ya calculados.
    vector_writer, vector_video_path, vector_video_fps = open_vector_video_writer(
        out_dir,
        width,
        height,
        fps,
        stride,
        vector_video_settings,
    )
    if vector_writer is not None:
        try:
            for result in pair_results:
                fa = result["frame_a"]
                fb = result["frame_b"]
                frame_a = video_uint8[fa] if video_uint8 is not None else read_frame(cap, fa)
                vector_frame = draw_vector_video_frame(
                    frame_a,
                    result["x"],
                    result["y"],
                    result["u_out"],
                    result["v_out"],
                    result["valid_out"],
                    roi,
                    mask_rects,
                    fa,
                    fb,
                    result["time_s"],
                    vector_video_settings,
                )
                vector_writer.write(vector_frame)
        finally:
            vector_writer.release()

    cap.release()

    summary = {
        "video": str(video_path),
        "fps": fps,
        "metadata_fps": metadata_fps,
        "fps_override": fps_override,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "dt_s": dt,
        "start_frame": start_frame,
        "calibration_frame": calibration_frame,
        "roi": roi,
        "mask_rects": mask_rects,
        "mask_polygons": mask_polygons,
        "passes": [cfg.__dict__ for cfg in passes],
        "pairs_requested": pairs_config,
        "pairs_processed": pairs_done,
        "stride_requested": stride_requested,
        "stride": stride,
        "overlay_every": overlay_every,
        "write_px_csv": write_px_csv,
        "parallel": {
            "enabled": parallel_enabled,
            "workers": workers,
            "chunksize": chunksize,
            "frame_cache_size": frame_cache_size,
            "settings": parallel_settings,
        },
        "temporal_filter": temporal_filter_settings,
        "vector_video": {
            "enabled": bool(vector_video_settings.get("enabled", False)),
            "path": str(vector_video_path) if vector_video_path else None,
            "fps": vector_video_fps,
            "settings": vector_video_settings,
        },
        "min_corr": min_corr,
        "local_threshold_px": local_threshold_px,
        "spatial_filter_enabled": spatial_filter_enabled,
        "correlation_method": correlation_method,
        "fft": {
            "enabled": correlation_method == "fft",
            "passes": [
                {
                    "window": cfg.window,
                    "overlap": cfg.overlap,
                    "step": max(1, int(round(cfg.window * (1.0 - cfg.overlap)))),
                    "padding": fft_padded_size(cfg.window, cfg.search),
                    "search": cfg.search,
                }
                for cfg in passes
            ],
            "mean_subtraction": True,
            "hann_windows": [[cfg.window, cfg.window] for cfg in passes],
            "batched_rfft2": True,
            "batched_irfft2": True,
            "batch_size": FFT_BATCH_SIZE,
            "frame_loading": "on_demand_cache",
        },
        "calibration": calibration,
        "vectors_total": vectors_total,
        "vectors_valid": valid_count,
        "valid_fraction": valid_count / vectors_total if vectors_total else 0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
