# pRV modelling with Gaussian Process

Demo for the NIRPS data week.

The examples for the demo are available as Python script (with "percent" ipython cells) and as Jupyter Notebooks (generated with `jupytext`).
The scripts are in `scripts` and the notebooks are in `notebooks`.

## Quick look and preprocessing

The first notebook (`00_quicklook.ipynb`) simply generates a few plots and performs simple preprocessing.
It also removes some outliers and saves a new RDB file.

## RV+activity modelling

The RV modelling demo uses well tested tools that have been in use for a long time,
such as `george`, `celerite` and `radvel`. Two of them are in maintenance mode,
but they are still used widely, including by some higher-level packages such as
`juliet`.

To install the requirements for this example, use `python -m pip install -r requirements.txt`

The notebook is `01_george_radvel.ipynb`
