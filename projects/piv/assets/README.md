# Provenance of selected figures

The files were selected from the Drive folder supplied by Matías for the portfolio. No material from the older local Desktop copy was used.

- `experimental-setup.png`: original `Presentación/Informe/Set_up.png`, downloaded without modification.
- `input-pair-a.png` and `input-pair-b.png`: original `Presentación/Informe/img/metodologia/piv_par_A.png` and `piv_par_B.png`. These illustrate the image-pair method in the thesis presentation and are not asserted to be pair 300→304 from the displayed result.
- `velocity-example.png`: frame at 10 s of playback from `PROCESAR PIV HP/reprocesados/ensayo_olas_60grados_1/piv_manual/piv_fft_campo_velocidad_savgol_7x7_o3.mp4`.
- `vorticity-example.png`: corresponding playback position from `piv_fft_vorticidad_savgol_7x7_o3_sigma3.mp4` in the same directory.
- `velocity-90deg-t2.562s.png`: frame pair 615→619 at t = 2.562 s from the 90° velocity sequence in Drive. The frame was captured from the displayed video and cropped to the video frame only; no field data, labels or colour limits were changed.
- `vorticity-sequence-90deg.webp`: nine-frame 90° vorticity montage bundled from the Drive preview at 1600 px. The panel order, time labels and shared colour scale are preserved; the plotted data were not recomputed or recoloured.

The earlier 60° result images retain the original overlay labelling pair 300→304 and physical time 1.250 s. They were decoded directly to PNG with FFmpeg, with no cropping, recolouring or numerical-field recomputation. Their legends, masks and annotations are preserved.

The original run and rendering summaries are the basis of `../processing-record.json`. Per-computer paths are omitted from that portable record. The two summaries report different aggregate valid-vector fractions; this portfolio does not merge them into an accuracy claim.
