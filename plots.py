import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

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
    arrival_counts = list(np.sum(arrivals, axis=1))
    new_arrivals = [cur - prev for prev, cur in zip([0] + arrival_counts, arrival_counts)]
    arrival_years = [year for year, count in zip(common.YEARS, new_arrivals) for _ in range(count)]

    plot_fig, plot_ax = plt.subplots()
    sns.histplot(arrival_years, kde=True, ax=plot_ax, stat='probability', binwidth=1)

    median_label = '>2100' if np.isnan(median_arrival) else f'median ({median_arrival:.0f})'
    plt.axvline(median_arrival, c='red', linestyle='dashed', label=median_label)

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
