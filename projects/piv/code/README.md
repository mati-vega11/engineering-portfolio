# PIV research-source snapshot

These files are selected from the thesis project's `PROCESAR PIV HP` directory. Their algorithms are preserved as supplied; they have not been refactored or executed as part of preparing this portfolio.

| Source | Role |
| --- | --- |
| [procesar_piv.py](procesar_piv.py) | Image preparation, multi-pass direct/FFT correlation, subpixel estimation, vector quality checks and physical-unit export |
| [finalizar_piv_video_optimizado.py](finalizar_piv_video_optimizado.py) | Reconstruct compact fields and render flow quantities |
| [generar_video_vorticidad.py](generar_video_vorticidad.py) | Vorticity calculation, filtering and visualisation helpers |
| [savgol_piv_video_desde_compacto.py](savgol_piv_video_desde_compacto.py) | Separable Savitzky–Golay post-processing and velocity/vorticity videos |
| [regenerar_vorticidad_savgol_limite.py](regenerar_vorticidad_savgol_limite.py) | Re-render vorticity using selected visual limits |

## Configuration and use

Use Python 3.10 or newer. The supplied dependency list is unpinned, matching the research project's list; it is not an archived software environment.

```sh
python -m venv .venv
```

Activate that environment using the command appropriate to the operating system, then:

```sh
python -m pip install -r requirements.txt
python procesar_piv.py --config example_config.json --video /path/to/recording.mp4 --out results/my_run
```

The source recording is not bundled. The example configuration preserves the documented 60° run's numerical settings but uses a portable input filename and a short subset of image pairs. It requires the matching recording and calibration; it is not a universal configuration.

Before using another recording, set its physical acquisition rate, image-pair separation, ROI, masks and spatial scale. Video-file playback FPS can differ from camera acquisition FPS.

The scripts in this folder depend on one another, so preserve their relative locations. A complete run and its subsequent post-processing require the corresponding source video and intermediate data; those large research datasets are not included here.

## Interpretation

- A correlation threshold is a vector-quality criterion, not a calibrated velocity error bound.
- The example disables the optional local spatial median validator; do not describe that validator as active for this run.
- The second PIV pass uses a displacement predictor. It does not perform interrogation-window deformation.
- A fixed colour limit changes display saturation, not the underlying vorticity values.
- Keep physical acquisition time separate from slow-motion playback time.
