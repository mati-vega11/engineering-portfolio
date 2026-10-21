# Provenance of selected figures

The files were selected from the Drive folder supplied by Matías for the portfolio. No material from the older local Desktop copy was used.

- `experimental-setup.png`: original `Presentación/Informe/Set_up.png`, downloaded without modification.
- `input-pair-a.png` and `input-pair-b.png`: original `Presentación/Informe/img/metodologia/piv_par_A.png` and `piv_par_B.png`. These illustrate the image-pair method in the thesis presentation; they are not asserted to be pair 300→304 from the 90° result.
- `velocity-example.png`: frame at 10 s of playback from `PROCESAR PIV HP/reprocesados/ensayo_olas_60grados_1/piv_manual/piv_fft_campo_velocidad_savgol_7x7_o3.mp4`.
- `vorticity-example.png`: corresponding playback position from `piv_fft_vorticidad_savgol_7x7_o3_sigma3.mp4` in the same directory. These are supporting 60° examples, separate from the 90° comparison.
- `velocity-vorticity-90deg-t1.250s.png`: the current paired 90° result. Both panels are frame pair 300→304 at t = 1.250 s, decoded from the original Drive videos at 10.0 s playback. The plot fields and source colour scales were not recomputed or recoloured. The composition crops video margins and the baked-in top text band, retains the original colour bars, and adds 1.36× insets around the plate. The velocity panel reproduces the source overlay’s 93.7% valid-vector value; the vorticity panel retains its ±9 s⁻¹ saturated display range, σ = 3 vector-node Gaussian smoothing and |ωz| < 1 s⁻¹ mask.
- `velocity-90deg-t2.562s.png`: an earlier velocity-only selection, retained from a previous draft and no longer featured.
- `vorticity-sequence-90deg.webp`: nine-frame 90° vorticity montage bundled from the Drive preview at 1600 px. The panel order, time labels and shared colour scale are preserved; the plotted data were not recomputed or recoloured.

The two 60° example images preserve their own original overlays, masks and scales. They are not merged with the 90° results, and this portfolio does not treat the valid-vector fraction as a measurement-accuracy claim.

The original run and rendering summaries are the basis of `../processing-record.json`. Per-computer paths are omitted from that portable record. The two summaries report different aggregate valid-vector fractions, which are kept separate.
