import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools

import common


def plot_timeline(tl: common.Timeline, y_lab: str, errorbar_interval: int = 95):
    plot_fig, plot_ax = plt.subplots()

    rotated = np.vstack([np.array(list(zip(itertools.repeat(idx + common.START_YEAR), tl[:, idx])))
                         for idx in common.YEAR_OFFSETS])
    df = pd.DataFrame({
        'Year': rotated[:, 0],
        y_lab: 10 ** rotated[:, 1]
    })
    sns.lineplot(df, x='Year', y=y_lab, errorbar=('pi', errorbar_interval), estimator='median', ax=plot_ax)
    plot_ax.set_yscale('log')
    return plot_fig


def plot_tai_timeline(tai_timeline):
    # TODO: take the raw arrival yes/nos, convert to a beta distribution for each year, use that to plot uncertainty?
    plot_fig, plot_ax = plt.subplots()

    sns.lineplot(pd.DataFrame({
        'Year': np.arange(common.START_YEAR, common.END_YEAR),
        'P(TAI)': tai_timeline,
    }), x='Year', y='P(TAI)', ax=plot_ax)

    return plot_fig


def plot_tai_timeline_density(arrivals):
    arrival_counts = list(np.sum(arrivals, axis=1))
    new_arrivals = [cur - prev for prev, cur in zip([0] + arrival_counts, arrival_counts)]
    arrival_years = [year for year, count in zip(common.YEARS, new_arrivals) for _ in range(count)]

    plot_fig, plot_ax = plt.subplots()
    sns.histplot(arrival_years, kde=True, ax=plot_ax, stat='probability', binwidth=1)

    return plot_fig


def plot_tai_requirements(tai_requirements: common.Distribution, x_lab: str):
    g = sns.displot(tai_requirements, kde=True)
    g.set_xlabels(x_lab)
    return g
