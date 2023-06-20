import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import rv_continuous

import common


def plot_timeline(tl: common.Timeline, y_lab: str, errorbar_interval: int = 90) -> Figure:
    plot_fig, plot_ax = plt.subplots()

    rotated = np.vstack([np.array(list(zip(itertools.repeat(idx + common.START_YEAR), tl[:, idx])))
                         for idx in common.YEAR_OFFSETS])
    df = pd.DataFrame({
        'Year': rotated[:, 0],
        y_lab: 10**rotated[:, 1]
    })
    sns.lineplot(df, x='Year', y=y_lab, errorbar=('pi', errorbar_interval), estimator='median', ax=plot_ax)
    plot_ax.set_yscale('log')
    return plot_fig


def plot_tai_timeline(tai_timeline, median_arrival: float, x_lab: str, y_lab: str, title: str) -> Figure:
    plot_fig, plot_ax = plt.subplots()

    sns.lineplot(pd.DataFrame({
        'Year': np.arange(common.START_YEAR, common.END_YEAR),
        'P(TAI)': tai_timeline,
    }), x=x_lab, y=y_lab, ax=plot_ax)

    median_label = '>2100' if np.isnan(median_arrival) else f'median ({median_arrival:.0f})'
    plt.axvline(median_arrival, c='red', linestyle='dashed', label=median_label)

    plt.legend()
    plot_ax.set_title(title)

    return plot_fig


def plot_tai_timeline_density(arrivals, median_arrival: float, x_lab: str, y_lab: str, title: str) -> Figure:
    samples = arrivals.shape[1]
    arrival_counts = list(np.sum(arrivals, axis=1))
    new_arrivals = np.array([cur - prev for prev, cur in zip([0] + arrival_counts, arrival_counts)])
    arrival_years = [year for year, count in zip(common.YEARS, new_arrivals) for _ in range(count)]

    plot_fig, plot_ax = plt.subplots()
    sns.histplot(arrival_years, kde=True, ax=plot_ax, stat='probability', binwidth=1)

    median_label = '>2100' if np.isnan(median_arrival) else f'median ({median_arrival:.0f})'
    plt.axvline(median_arrival, c='red', linestyle='dashed', label=median_label)

    # Box to highlight the amount of density that's truncated
    missing_density = 1 - arrival_counts[-1]/samples
    max_density_under_box = max(new_arrivals[2072-2023:] / samples)
    plot_ax.text(2072, max_density_under_box + 0.01, f"P(>2100) = {missing_density:.1%}\n", linespacing=0.15,
                 bbox=dict(boxstyle="round, rounding_size=0.2", facecolor='white', alpha=0.1, edgecolor='black'))
    plot_ax.arrow(2088.2, max_density_under_box + 0.0087, dx=8, dy=0, color='#666',
                  width=0.0010, head_width=0.0028, head_length=2.5)

    plt.legend()
    plot_ax.set_xlabel(x_lab)
    plot_ax.set_ylabel(y_lab)
    plot_ax.set_title(title)

    return plot_fig


def plot_tai_requirements(tai_requirements: common.Distribution, x_lab: str,
                          title: str, cumulative: bool = False) -> Figure:
    plot_fig, plot_ax = plt.subplots()
    sns.histplot(tai_requirements, kde=True, ax=plot_ax, stat='probability', binwidth=1, cumulative=cumulative)
    plot_ax.set_xlabel(x_lab)
    plot_ax.set_title(title)

    return plot_fig


def plot_tai_requirements_adjustment_decomposition(prior: rv_continuous, upper_bound: rv_continuous, posterior: rv_continuous,
        x_lab: str, title: str, cumulative: bool = False) -> Figure:
    plot_fig, plot_ax = plt.subplots()

    # TODO Automatically compute limits in X
    grid = np.linspace(10, 100, 500)

    df = lambda dist: dist.cdf if cumulative else dist.pdf
    sns.lineplot(x=grid, y=df(prior)(grid),       ax=plot_ax, label="Prior")
    sns.lineplot(x=grid, y=df(upper_bound)(grid), ax=plot_ax, label="Upper bound")
    sns.lineplot(x=grid, y=df(posterior)(grid),   ax=plot_ax, label="Posterior", linestyle='--')

    plot_ax.legend()
    plot_ax.set_xlabel(x_lab)
    plot_ax.set_title(title)

    return plot_fig
