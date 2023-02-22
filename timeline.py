import numpy as np
import numpy.typing as npt
from common import DistributionCI, Timeline, Distribution, NUM_SAMPLES, YEAR_OFFSETS
from typing import List
import gpu_efficiency
from k_performance import computation_for_k_performance


def spending(
    growth_rate: DistributionCI = DistributionCI('normal', 70, 1.1480341, 3.278781),
    gwp_growth_rate: DistributionCI = DistributionCI('normal', 70, 0.0114741, 0.045136),
    maximum_gwp_percentage: DistributionCI = DistributionCI('normal', 70, 0.0000083, 0.014047),
    starting_gwp: float = 1e14,
    starting_max_spend: float = 20e6,
) -> Timeline:
    """
    We assume that the current maximum amount people are willing to spend on a training run is $20e6, and that it will
    grow at `growth_rate` until we reach `maximum_gwp_percentage` of GWP, at which point it will grow at the rate of
    GWP. Current GWP is 1e14.

    TODO: improve numbers by taking a look at Ben's estimates, and gdoc where I asked for input
    """
    growth_rate_samples = np.maximum(growth_rate.sample(NUM_SAMPLES), 0.0000001)  # TODO: hack
    maximum_gwp_percentage_samples = np.maximum(maximum_gwp_percentage.sample(NUM_SAMPLES), 0.000001)  # TODO: hack
    gwp_growth_rate_samples = gwp_growth_rate.sample(NUM_SAMPLES)

    spending_rollouts = []
    for i in range(NUM_SAMPLES):
        spending_rollouts.append([])
        gwp = np.log10(starting_gwp)
        max_spend = np.log10(starting_max_spend)
        for _ in YEAR_OFFSETS:
            max_spend += np.log10(1 + growth_rate_samples[i])
            gwp += np.log10(1 + gwp_growth_rate_samples[i])
            dollar_limit = gwp + np.log10(maximum_gwp_percentage_samples[i])
            spending_rollouts[i].append(dollar_limit * (1 - np.exp(-max_spend/dollar_limit)))

    return np.stack(spending_rollouts)


def flops_per_dollar(
    transistors_per_core_limit: DistributionCI = DistributionCI('lognormal', 70, 0.896, 1.979),
    process_size_limit: DistributionCI = DistributionCI('lognormal', 70, 1.396, 2.479),
    process_efficiency: DistributionCI = DistributionCI('lognormal', 95, 0.005, 0.01),
    hardware_specialization: DistributionCI = DistributionCI('lognormal', 95, 0.005, 0.01),
    gpu_dollar_cost: int = 500,
):
    transistors_per_core_limit_samples = transistors_per_core_limit.sample(NUM_SAMPLES)
    process_size_limit_samples = process_size_limit.sample(NUM_SAMPLES)
    process_efficiency_samples = process_efficiency.sample(NUM_SAMPLES)
    hardware_specialization_samples = hardware_specialization.sample(NUM_SAMPLES)

    # We use Marius's estimate to get a baseline projection, as a list of rollouts
    log_flops_per_second: List[List[float]] = gpu_efficiency.baseline_flops_per_second(
        NUM_SAMPLES, process_size_limit_samples, transistors_per_core_limit_samples
    )

    # Then, we modify the baseline with hardware_specialization, and process_efficiency (the latter only kicking in
    # once the rate of improvement falls below 10% per year)
    log_flops_per_dollar = []
    rollout_years = len(log_flops_per_second[0])
    for rollout_idx, rollout in enumerate(log_flops_per_second):
        log_flops_per_dollar.append([])
        cumulative_multiplier = 1
        process_efficiency_rate = 1
        hardware_specialization_rate = 1 + hardware_specialization_samples[rollout_idx]
        for year_offset in YEAR_OFFSETS:
            cumulative_multiplier *= hardware_specialization_rate * process_efficiency_rate
            if year_offset < rollout_years:
                rollout[year_offset] += np.log10(cumulative_multiplier)
            else:
                # We've now progressed past the years considered in the baseline projection,
                # so use the previous year as the baseline
                rollout.append(rollout[-1] + np.log10(cumulative_multiplier))

            # check if rate of improvement has fallen below 10%,
            # at which point we start considering process_efficiency
            # TODO: does this do what we want when year_offset is 0?
            growth_rate = 10**(rollout[year_offset] - rollout[year_offset - 1]) - 1 if year_offset else 0
            if year_offset and growth_rate < np.log10(1.1):
                process_efficiency_rate = 1 + process_efficiency_samples[rollout_idx]

            amortization_years = 1.2 / (growth_rate + 0.1)
            amortization_seconds = 365 * 24 * 60 * 60 * amortization_years
            log_flops_per_dollar[rollout_idx].append(rollout[year_offset] +
                                                     np.log10(amortization_seconds / gpu_dollar_cost))
    return np.stack(log_flops_per_dollar)


def algorithmic_improvements(
    growth_rate: DistributionCI = DistributionCI('normal', 95, .246, 2.1518),
    transfer_multiplier: DistributionCI = DistributionCI('normal', 70, 0.4, 1.1),
    limit: DistributionCI = DistributionCI('lognormal', 70, 1e2, 1e10),
) -> Timeline:
    """
    Three components:
    - Base growth rate, from the "Algorithmic Progress in Computer Vision" paper gives us 100.96% each year, 95% CI is [24.60%, 215.18%]. We could combine this with Anson's findings on algo progress in LMs, if that becomes available. Should probably include some negative values (regulation, knowledge loss, etc)
    - Domain transfer multiplier: how much we should modify the rate to account for algorithmic progress being a different domain.
    - Limit: at what multiplier does algorithmic growth stop?

    Growth slows as we approach the limit. Distribution values represent the quantity you should multiply physical compute by to get effective compute
    """
    tai_growth_rate_samples = growth_rate.sample(NUM_SAMPLES) * transfer_multiplier.sample(NUM_SAMPLES)
    limit_samples = np.log10(np.maximum(limit.sample(NUM_SAMPLES), 0.0001))  # TODO: hack

    algorithmic_improvement = []
    for rollout in range(NUM_SAMPLES):
        algorithmic_improvement.append([])
        rate = 0
        for _ in YEAR_OFFSETS:
            rate += np.log10(1 + tai_growth_rate_samples[rollout])
            algorithmic_improvement[rollout].append(limit_samples[rollout] *
                                                    (1 - np.exp(-rate / limit_samples[rollout])))
    return np.stack(algorithmic_improvement)


def tai_requirements(
    slowdown: DistributionCI = DistributionCI('lognormal', 70, 9.85, 289.05),
    log_k_performance: DistributionCI = DistributionCI('lognormal', 70, 3129, 141714),
) -> Distribution:
    """
    User specifies:
    - slowdown: the degree to which the human judge will update slower than the ideal predictor.
    - k_performance: the length of the transformative task

    TODO: what's going on with the unused uncertainty over A,B,alpha,beta params in Matthew's notebook?
    """
    slowdown_samples = slowdown.sample(NUM_SAMPLES)
    log_k_performance_samples = log_k_performance.sample(NUM_SAMPLES)

    log_flops_needed_sample = []
    for i in range(len(slowdown_samples)):
        c = computation_for_k_performance(log_k_performance_samples[i], slowdown_samples[i])
        log_flops_needed_sample.append(c)
    return np.array(log_flops_needed_sample)


def sample_timeline():
    compute_available = spending() + flops_per_dollar() + algorithmic_improvements()
    arrivals = compute_available.T > tai_requirements()

    return np.sum(arrivals, axis=1) / NUM_SAMPLES


if __name__ == '__main__':
    print('running')
    #print(sample_timeline())
    print(algorithmic_improvements())