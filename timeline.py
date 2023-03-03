import numpy as np
import numpy.typing as npt
from common import DistributionCI, Timeline, Distribution, NUM_SAMPLES, YEAR_OFFSETS, constrain
from typing import List
import gpu_efficiency
from k_performance import computation_for_k_performance


def spending(
    samples: int = NUM_SAMPLES,
    growth_rate: DistributionCI = DistributionCI('normal', 70, 1.1480341, 3.278781),
    gwp_growth_rate: DistributionCI = DistributionCI('normal', 70, 0.0114741, 0.045136),
    maximum_gwp_percentage: DistributionCI = DistributionCI('normal', 70, 0.0000083, 0.014047),
    starting_gwp: float = 1e14,
    starting_max_spend: float = 2e7,
) -> Timeline:
    """
    We assume that the current maximum amount people are willing to spend on a training run is $2e7, and that it will
    grow at `growth_rate` until we reach `maximum_gwp_percentage` of GWP, at which point it will grow at the rate of
    GWP.

    TODO: improve numbers by taking a look at Ben's estimates
    """
    # Make sure the growth multiplier is positive
    growth_rate_samples = np.maximum(growth_rate.sample(samples), -1 + 1e-10)
    # TODO: change GWP percentage to a beta?
    maximum_gwp_percentage_samples = np.maximum(maximum_gwp_percentage.sample(samples), 1e-10)
    gwp_growth_rate_samples = gwp_growth_rate.sample(samples)

    spending_rollouts = []
    for i in range(samples):
        spending_rollouts.append([])
        gwp = np.log10(starting_gwp)
        max_spend = np.log10(starting_max_spend)
        for _ in YEAR_OFFSETS:
            max_spend += np.log10(1 + growth_rate_samples[i])
            gwp += np.log10(1 + gwp_growth_rate_samples[i])
            dollar_limit = gwp + np.log10(maximum_gwp_percentage_samples[i])
            spending_rollouts[i].append(constrain(value=max_spend, limit=dollar_limit))

    return np.stack(spending_rollouts)


def flops_per_dollar(
    samples: int = NUM_SAMPLES,
    transistors_per_core_limit: DistributionCI = DistributionCI('lognormal', 70, 0.896, 1.979),
    process_size_limit: DistributionCI = DistributionCI('lognormal', 70, 1.396, 2.479),
    process_efficiency: DistributionCI = DistributionCI('lognormal', 95, 0.005, 0.01),
    hardware_specialization: DistributionCI = DistributionCI('lognormal', 95, 0.005, 0.01),
    gpu_dollar_cost: int = 500,
):
    transistors_per_core_limit_samples = transistors_per_core_limit.sample(samples)
    process_size_limit_samples = process_size_limit.sample(samples)
    process_efficiency_samples = process_efficiency.sample(samples)
    hardware_specialization_samples = hardware_specialization.sample(samples)

    # We use Marius's estimate to get a baseline projection, as a list of rollouts
    log_flops_per_second: List[List[float]] = gpu_efficiency.baseline_flops_per_second(
        samples, process_size_limit_samples, transistors_per_core_limit_samples
    )

    # Then, we modify the baseline with hardware_specialization and process_efficiency (the latter only kicking in
    # once the rate of improvement falls below 10% per year)
    log_flops_per_dollar = []
    for rollout_idx, rollout in enumerate(log_flops_per_second):
        log_flops_per_dollar.append([])
        cumulative_multiplier = 1
        process_efficiency_rate = 1
        hardware_specialization_rate = 1 + hardware_specialization_samples[rollout_idx]
        for year_offset in YEAR_OFFSETS:
            cumulative_multiplier *= hardware_specialization_rate * process_efficiency_rate
            rollout[year_offset] += np.log10(cumulative_multiplier)

            # check if rate of improvement has fallen below 10%, at which point we start considering process_efficiency
            # Use a reasonable, >10% growth rate for the first year
            growth_rate = 10**(rollout[year_offset] - rollout[year_offset - 1]) - 1 if year_offset else 0.37
            if year_offset and growth_rate < 0.1:
                process_efficiency_rate = 1 + process_efficiency_samples[rollout_idx]

            amortization_years = 1.2 / (growth_rate + 0.1)
            amortization_seconds = 365 * 24 * 60 * 60 * amortization_years
            log_flops_per_dollar[rollout_idx].append(rollout[year_offset] +
                                                     np.log10(amortization_seconds / gpu_dollar_cost))
    return np.stack(log_flops_per_dollar)


def algorithmic_improvements(
    growth_rate: DistributionCI = DistributionCI('normal', 95, 0.246, 2.1518),
    transfer_multiplier: DistributionCI = DistributionCI('normal', 70, 0.4, 1.1),
    limit: DistributionCI = DistributionCI('lognormal', 70, 1e2, 1e10),
    samples: int = NUM_SAMPLES,
):
    """
    Three components:
    - Base growth rate, from the "Algorithmic Progress in Computer Vision" paper gives us 100.96% each year, 95% CI is
    [24.60%, 215.18%]. We could combine this with Anson's findings on algo progress in LMs, if that becomes available.
    - Domain transfer multiplier: how much we should modify the rate to account for algorithmic progress being a
    different domain.
    - Limit: at what multiplier does algorithmic growth stop?

    Growth slows as we approach the limit. Distribution values represent the quantity you should multiply physical
    compute by to get effective compute
    """
    # Algorithmic regress is possible (due to regulation, knowledge loss, eg), which is represented by a multiplier
    # between 0 and 1. But a negative rate is not possible.
    transfer_multiplier_samples = np.maximum(transfer_multiplier.sample(samples), 0)
    growth_multiplier_samples = np.maximum(1 + (growth_rate.sample(samples) * transfer_multiplier_samples), 1e-10)
    # On the other hand, current performance means that we know the lower limit for the multiplier is at least 1, so we
    # do enforce that
    limit_samples = np.log10(np.maximum(limit.sample(samples), 1))

    algorithmic_improvement = []
    for rollout in range(samples):
        algorithmic_improvement.append([])
        current_improvement = 0
        for _ in YEAR_OFFSETS:
            current_improvement += np.log10(growth_multiplier_samples[rollout])
            algorithmic_improvement[rollout].append(constrain(value=current_improvement, limit=limit_samples[rollout]))

    return np.stack(algorithmic_improvement)


def tai_requirements(
    samples: int = NUM_SAMPLES,
    slowdown: DistributionCI = DistributionCI('lognormal', 70, 9.85, 289.05),
    log_k_performance: DistributionCI = DistributionCI('lognormal', 70, 3129, 141714),
) -> Distribution:
    """
    User specifies:
    - slowdown: the degree to which the human judge will update slower than the ideal predictor.
    - k_performance: the length of the transformative task

    TODO: what's going on with the unused uncertainty over A,B,alpha,beta params in Matthew's notebook?
    """
    slowdown_samples = slowdown.sample(samples)
    log_k_performance_samples = log_k_performance.sample(samples)

    log_flops_needed_sample = []
    for i in range(len(slowdown_samples)):
        c = computation_for_k_performance(log_k_performance_samples[i], slowdown_samples[i])
        log_flops_needed_sample.append(c)
    return np.array(log_flops_needed_sample)


def sample_timeline(
    samples: int = NUM_SAMPLES,
):
    compute_available = spending(samples=samples) + flops_per_dollar(samples=samples) + algorithmic_improvements(samples=samples)
    arrivals = compute_available.T > tai_requirements(samples=samples)

    return np.sum(arrivals, axis=1) / samples


if __name__ == '__main__':
    print('running')
    #print(sample_timeline())
    print(algorithmic_improvements())