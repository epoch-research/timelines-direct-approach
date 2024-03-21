import timeline
import numpy as np
from common import DistributionCI, YEAR_OFFSETS, NUM_SAMPLES
from app import SPENDING_PLOT_PARAMS
from plots import plot_timeline, PlotFormat, reformat
import matplotlib.pyplot as plt
import pandas as pd
import common
import os


os.makedirs('output', exist_ok=True)


def spending_plot():
    np.random.seed(0)

    # Start at 2024
    years = 2024 + np.array(common.YEAR_OFFSETS)

    gemini_compute = DistributionCI('lognormal', 95, 1.1e+25, 2.1e+26).change_width().sample(NUM_SAMPLES) # FLOP
    gemini_price_per_hour = DistributionCI('lognormal', 90, 1.89, 5.40).change_width().sample(NUM_SAMPLES) # $/hour
    gemini_compute_per_hour = 3600 * 9.9e14 * DistributionCI('lognormal', 90, 0.2, 0.4).change_width().sample(NUM_SAMPLES) # FLOP/h
    starting_max_spend_samples = gemini_compute / gemini_compute_per_hour * gemini_price_per_hour
    starting_max_spend_samples /= 1e6 # in million dollars

    median = np.quantile(DistributionCI('lognormal', 90, 1.5e8/1e6, 1.7e9/1e6).change_width().sample(NUM_SAMPLES), 0.5)
    print(median)

    spending_params = {
        'samples': NUM_SAMPLES,
        'starting_gwp': 1.17e14,
        'starting_max_spend': DistributionCI('lognormal', 90, 1.5e8/1e6, 1.7e9/1e6).change_width(),
        'invest_growth_rate': DistributionCI('normal', 90, 180, 270).change_width(),
        'gwp_growth_rate': DistributionCI('normal', 80, 0.4, 4).change_width(),
        'max_gwp_pct': DistributionCI('lognormal', 80, 0.01, 2).change_width(),
        'begin_at_initial_spending': True,
    }

    investment_timeline = timeline.spending(**spending_params)

    q_05 = 10**np.quantile(investment_timeline, 0.05, axis=0)
    q_50 = 10**np.quantile(investment_timeline, 0.50, axis=0)
    q_95 = 10**np.quantile(investment_timeline, 0.95, axis=0)

    # Plot
    fig = plt.figure()
    plt.plot(years, q_50)
    plt.fill_between(years, q_05, q_95, alpha=0.2)

    plt.yscale('log')
    plt.ylabel('Largest Training Run ($)')
    plt.xlabel('Year')
    plt.title('Cost of Training Frontier Machine Learning Models')
    plt.xlim(2024 - 0.1, 2030)

    suffixes = {
        3: 'K',
        6: 'M',
        9: 'B',
        12: 'T',
        15: 'E',
    }

    tick_labels = []
    ticks = plt.yticks()[0]
    for tick in ticks:
        order = 3 * np.floor(np.log10(tick)/3)
        tick_labels.append(f'{int(tick/10**order)}{suffixes[order]}')

    plt.yticks(ticks, tick_labels)

    plt.ylim(1e8, 1e12)

    # Save as regular figure
    fig.savefig('output/spending.svg')

    # Save as Epoch figure
    epoch_fig = reformat(fig, PlotFormat.EPOCH)
    epoch_fig.caption('Extrapolated inflation-adjusted hardware purchase costs of the computational resources needed to train frontier machine learning models, based on the data from 125 systems between 2010 and 2023.')
    epoch_fig.y_ticks(ticks, tick_labels)
    epoch_fig.export(format='figure', filename='output/spending.json')

    # Save to CSV
    clip_index = np.where(years == 2030)[0][0]
    df = pd.DataFrame({'year': years[:clip_index+1], 'q_05': q_05[:clip_index+1], 'q_50': q_50[:clip_index+1], 'q_95': q_95[:clip_index+1]})
    df.to_csv('output/spending.csv', index=False)


if __name__ == '__main__':
    spending_plot()
