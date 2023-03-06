import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools

import common
import timeline


def plot_timeline(tl: common.Timeline, y_lab: str, errorbar_interval: int = 95):
    plot_fig, plot_ax = plt.subplots()

    rotated = np.vstack([np.array(list(zip(itertools.repeat(idx + common.START_YEAR), tl[:, idx])))
                         for idx in common.YEAR_OFFSETS])
    df = pd.DataFrame({
        'Year': rotated[:, 0],
        y_lab: rotated[:, 1]
    })
    sns.lineplot(df, x='Year', y=y_lab, errorbar=('pi', errorbar_interval), estimator='median', ax=plot_ax)
    return plot_fig


def plot_tai_timeline(tai_timeline):
    # TODO: it'd be interesting to make this take the raw arrival yes/nos, convert to a beta distribution for each year,
    #   and use that to plot uncertainty
    plot_fig, plot_ax = plt.subplots()

    sns.lineplot(pd.DataFrame({
        'Year': np.arange(common.START_YEAR, common.END_YEAR),
        'P(TAI)': tai_timeline,
    }), x='Year', y='P(TAI)', ax=plot_ax)

    return plot_fig

def plot_tai_timeline_density(tai_timeline):
    plot_fig, plot_ax = plt.subplots()



    g = sns.distplot(tai_timeline, kde=True)

    return plot_fig


def plot_tai_requirements(tai_requirements: common.Distribution, x_lab:str):
    g = sns.displot(tai_requirements, kde=True)
    g.set_xlabels(x_lab)
    return g
