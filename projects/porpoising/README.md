# Simplified Formula 1 porpoising model

**Numerical Methods course project | Universidad de Chile | Spring semester 2025**

[Back to portfolio](../../README.md)

## Engineering objective

Develop an academic reduced-order model to explore coupled vertical heave and pitch in an F1-style car. The model represents suspension forces, front and rear aerodynamic loads, and a floor load that changes with ride height, including a prescribed loss-of-ground-effect function to explore a porpoising-like response.

## My contribution

- Formulated and implemented the coupled heave/pitch model in Python and MATLAB.
- Represented the ride-height-dependent floor load and its prescribed stall behaviour.
- Integrated the equations of motion with fourth-order Runge–Kutta (RK4).
- Analysed vertical displacement and pitch responses with time histories and FFT-based frequency analysis.
- Presented the model and results as a Numerical Methods course project.

## Project files

- [Simulation notebook](porpoising_model.ipynb) — the notebook exported from Google Colab.
- [Course presentation](porpoising_model_presentation.pptx) — the 23-slide presentation in PowerPoint format.
- [Open the notebook in Google Colab](https://colab.research.google.com/drive/1BTo0xbO4PWdxj0lQWC-oeDabNm6ygjQz).

## Scope and reproducibility

This is an academic reduced-order model. The prescribed aerodynamic relationship is a modelling assumption used to explore a porpoising-like response; the project is not a CFD solution of underfloor flow and is not a validated prediction for a real Formula 1 vehicle.

The Colab notebook contains parameter changes and exploratory code added after the course presentation, so its current parameter values may not reproduce the results shown in the slides. The presentation records the version discussed for the course project; the notebook is shared as an evolving code artifact. Execution outputs and account-specific runtime metadata were cleared for publication, while the code cells were retained. The appended script cell is guarded so its command-line entry point does not run when the notebook is opened in Colab.

## Tools

Python, MATLAB, NumPy, Matplotlib, RK4 and FFT.
