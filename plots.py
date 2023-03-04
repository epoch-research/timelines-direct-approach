import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools

import common
import timeline


def plot(tl: common.Timeline, y_lab: str):
    plot_fig, plot_ax = plt.subplots()

    rotated = np.vstack([np.array(list(zip(itertools.repeat(idx + common.START_YEAR), tl[:, idx])))
                         for idx in common.YEAR_OFFSETS])
    df = pd.DataFrame({
        'Year': rotated[:, 0],
        y_lab: rotated[:, 1]
    })
    sns.lineplot(df, x='Year', y=y_lab, errorbar=('pi', 95), estimator='median', ax=plot_ax)
    return plot_fig
