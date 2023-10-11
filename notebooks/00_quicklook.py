# %% [markdown]
# #  Quick look at NIRPS proxima data

# %%
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.timeseries import LombScargle
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# %% [markdown]
# ## Loading the data

# %%
tbl = Table.read("./data/lbl_PROXIMA_PROXIMA.rdb")

rv_off = np.median(tbl["vrad"])

tbl["vrad"] -= rv_off

rjd_bjd_off = 2457000
extra_off_factor = 100
extra_off = np.floor(tbl["rjd"].min() / extra_off_factor) * extra_off_factor
t_off = rjd_bjd_off + extra_off

tlabel = "t"
tbl[tlabel] = tbl["rjd"] - extra_off

# %% [markdown]
# ## Timeseries and periodograms

# %%
qty_list = ["vrad", "d2v", "d3v", "DTEMP"]
qty_labels = {
    "vrad": "RV",
    "d2v": "D2V",
    "d3v": "D3V",
    "DTEMP": "$\Delta$T",
}
qty_units = {
    "vrad": "m/s",
    "d2v": "m$^2$ / s$^2$",
    "d3v": "m$^3$ / s$^3$",
    "DTEMP": "K",
}


# %%
def plot_timeseries(
    tbl: Table,
    qty_list: List[str],
    plot_ls: bool = True,
    ylog: bool = False,
    err_label: Optional[str] = None,
    err_fmt: str = "k.",
    target_fap: float = 0.01,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray[Axes]] = None,
) -> Tuple[Figure, Axes]:
    ncols = 2 if plot_ls else 1
    num_qty = len(qty_list)

    if fig is None:
        fig, axes = plt.subplots(
            nrows=num_qty,
            ncols=ncols,
            figsize=(12 * ncols, 3.5 * num_qty),
            sharex="col",
            squeeze=False,
        )
    elif axes is None:
        axes = np.array(fig.axes).reshape((num_qty, ncols))

    for i, qty in enumerate(qty_list):
        t = tbl[tlabel].data
        y = tbl[qty].data
        yerr = tbl[f"s{qty}"].data
        axes[i, 0].errorbar(t, y, yerr=yerr, fmt=err_fmt, label=err_label)
        axes[i, 0].set_ylabel(f"{qty_labels[qty]} [{qty_units[qty]}]")

        if plot_ls:
            ls = LombScargle(t, y, dy=yerr)
            fap = ls.false_alarm_level(target_fap)
            freq, power = ls.autopower(maximum_frequency=1.0)
            period = 1 / freq

            ls_window = LombScargle(
                t, np.ones_like(y), dy=yerr, fit_mean=False, center_data=False
            )
            power_window = ls_window.power(freq)

            axes[i, 1].plot(period, power, "k", label="Periodogram" if i == 0 else None)
            axes[i, 1].plot(
                period, power_window, "C1", label="Window Function" if i == 0 else None
            )
            axes[i, 1].axhline(
                fap,
                linestyle="--",
                color="r",
                label=f"{target_fap}% FA level" if i == 0 else None,
                alpha=0.5,
            )
            axes[i, 1].axvline(
                11.1888,
                linestyle="--",
                color="k",
                label="Confirmed planets b and d" if i == 0 else None,
                alpha=0.5,
            )
            axes[i, 1].axvline(5.167, linestyle="--", color="k", alpha=0.5)
            axes[i, 1].axvline(
                91,
                linestyle="--",
                color="C5",
                label="Rotation" if i == 0 else None,
                alpha=0.7,
            )
            axes[i, 1].axvline(
                1900,
                linestyle="--",
                label="Candidate Planet c" if i == 0 else None,
                alpha=0.5,
            )
            axes[i, 1].set_xscale("log")
            if ylog:
                axes[i, 1].set_yscale("log")
                axes[i, 1].set_ylim((1e-2, None))
            axes[i, 1].set_title(f"{qty_labels[qty]} LS Periodogram")
            axes[i, 1].set_ylabel("Power")
            # Get handles for periodograms only
            handles = []
            labels = []
            for ax in axes[:, 1].flatten():
                handles_sub, labels_sub = ax.get_legend_handles_labels()
                handles.extend(handles_sub)
                labels.extend(labels_sub)
            # Remove duplicate handles and labels
            handles, labels = zip(
                *sorted(set(zip(handles, labels)), key=lambda x: labels.index(x[1]))
            )

            fig.legend(handles, labels, loc="upper right")
    axes[-1, 0].set_xlabel(f"Time [BJD - {t_off:.0f}]")
    if plot_ls:
        axes[-1, 1].set_xlabel("Period [d]")

    return fig, axes


# %%
plot_timeseries(tbl, qty_list, plot_ls=True)
plt.show()

# %% [markdown]
# ## Outlier filtering


# %%
def sigma_clip_tbl(
    tbl: Table, qty_list: List[str], sigma: float = 5.0, **kwargs
) -> np.ndarray[bool]:
    mask = np.zeros(len(tbl), dtype=bool)
    for qty in qty_list:
        mask |= sigma_clip(tbl[qty], sigma=sigma, **kwargs).mask
    return mask


mask_sigma = sigma_clip_tbl(tbl, qty_list, sigma=5.0)

# %%
fig, axes = plot_timeseries(
    tbl[~mask_sigma], qty_list, plot_ls=False, err_label="Good points"
)
plot_timeseries(
    tbl[mask_sigma],
    qty_list,
    plot_ls=False,
    fig=fig,
    axes=axes,
    err_fmt="rx",
    err_label="Clipped points",
)
axes[0, 0].legend()
plt.show()


# %% [markdown]
# ## Error-based filtering


# %%
def plot_error_dist(
    tbl: Table, qty_list: List[str], quantile_cut: float = 0.95
) -> Tuple[Figure, Axes]:
    fig, axes = plt.subplots(ncols=len(qty_list), figsize=(24, 6))
    for i, qty in enumerate(qty_list):
        err_cut = np.quantile(tbl[f"s{qty}"], quantile_cut)
        axes[i].hist(tbl[f"s{qty}"], bins=50)
        axes[i].axvline(
            err_cut, color="r", linestyle="--", label=f"Quantile {quantile_cut}"
        )
        axes[i].set_title(f"{qty_labels[qty]} Error Histogram")
        axes[i].set_xlabel(f"{qty_labels[qty]} Error [{qty_units[qty]}]")
    axes[0].legend()
    return fig, axes


def error_quantile_clip(
    tbl: Table, qty_list: List[str], quantile_cut: float = 0.95
) -> np.ndarray[bool]:
    mask = np.zeros(len(tbl), dtype=bool)
    for qty in qty_list:
        err_cut = np.quantile(tbl[f"s{qty}"], quantile_cut)
        mask |= tbl[f"s{qty}"] > err_cut
    return mask


# %%
quantile_cut = 0.96
plot_error_dist(tbl, qty_list, quantile_cut)
plt.show()

mask_equant = error_quantile_clip(tbl, qty_list, quantile_cut)

# %%
mask = mask_sigma | mask_equant
fig, axes = plot_timeseries(tbl[~mask], qty_list, plot_ls=True, err_label="Good points")
plot_timeseries(
    tbl[mask_sigma],
    qty_list,
    plot_ls=False,
    fig=fig,
    axes=axes,
    err_fmt="rx",
    err_label="Sigma-clipped points",
)
plot_timeseries(
    tbl[mask_equant],
    qty_list,
    plot_ls=False,
    fig=fig,
    axes=axes,
    err_fmt="bx",
    err_label="Error-clipped points",
)
axes[0, 0].legend()
plt.show()

# %%
tbl.write("./data/lbl_PROXIMA_PROXIMA_preprocessed.rdb")
