import inspect
import pymc as pm
from typing import List, Callable, Union

import numpy as np
from scipy.stats import norm

import gpu_efficiency
from common import DistributionCI, Timeline, Distribution, NUM_SAMPLES, YEAR_OFFSETS, constrain, resample_between
from k_performance import computation_for_k_performance


def spending(
    samples: int = NUM_SAMPLES,
    # from Ben's estimate of 0.1 to 0.3 OOMs per year in https://epochai.org/blog/trends-in-the-dollar-training-cost-of-machine-learning-systems#appendix-i-overall-best-guess-for-the-growth-rate-in-training-cost:~:text=my%20all%2Dthings%2Dconsidered%20view%20is%200.2%20OOMs/year%20(90%25%20CI%3A%200.1%20to%200.3%20OOMs/year
    # For ref: Epoch staff aggregate: DistributionCI('normal', 70, 1.1480341, 3.278781)
    invest_growth_rate: DistributionCI = DistributionCI('normal', 90, 25.9, 99.5).change_width(),
    # 116_639_700 in 2023 -> 237_830_800 in 2060 -> 1.9% per year
    gwp_growth_rate: DistributionCI = DistributionCI('normal', 80, 0.4, 3.4).change_width(),
    # Ben's estimate (ie 0.004% to 5%) (Matthew's estimate: DistributionCI('lognormal', 70, 0.0000083, 0.014047))
    max_gwp_pct: DistributionCI = DistributionCI('lognormal', 95, 0.004, 5).change_width(),
    starting_gwp: float = 1.17e14,
    # Take 43 million for GPT-4 and increase by 0.2 OOMs. Use millions here to make it easier for users to enter values
    starting_max_spend: float = 4.3 * 10**0.2,
) -> Timeline:
    """
    We assume that the current maximum amount people are willing to spend on a training run is $2e7, which will grow at
    `invest_growth_rate` until we reach the `max_gwp_pct` of GWP, at which point it will grow at the rate of GWP
    (`gwp_growth_rate`).
    """
    # The growth rate can be negative, but the multiplier needs to be positive
    invest_growth_multiplier_samples = np.log10(resample_between(1 + (invest_growth_rate.sample(samples) / 100), min=0))
    max_gwp_pct_samples = np.log10(max_gwp_pct.sample(samples) / 100)
    # Again, the growth rate can be negative, but the multiplier needs to be positive
    gwp_growth_multiplier_samples = np.log10(resample_between(1 + (gwp_growth_rate.sample(samples) / 100), min=0))

    spending_rollouts = []
    for i in range(samples):
        spending_rollouts.append([])
        gwp = np.log10(starting_gwp)
        max_spend = np.log10(starting_max_spend * 1e6)  # convert to millions
        for _ in YEAR_OFFSETS:
            max_spend += invest_growth_multiplier_samples[i]
            gwp += gwp_growth_multiplier_samples[i]
            dollar_limit = gwp + max_gwp_pct_samples[i]
            constrained = constrain(value=max_spend, limit=dollar_limit)
            spending_rollouts[i].append(np.log10(max(1e-10, constrained)))

    return np.stack(spending_rollouts)


def flops_per_dollar(
    samples: int = NUM_SAMPLES,
    transistors_per_core_limit: DistributionCI = DistributionCI('lognormal', 70, 0.896, 1.98).change_width(),
    process_size_limit: DistributionCI = DistributionCI('lognormal', 70, 1.4, 2.48).change_width(),
    # expressed in OOMs
    hardware_specialization: DistributionCI = DistributionCI('lognormal', 90, 0.013, 0.176).change_width(),
    # Rough median cost of GPU with performance of between 1e13 and 1e14 FLOP/s (Projections are ~1e13.5 in 2023)
    gpu_dollar_cost: int = 1000,
):
    """
    General idea: Marius's projections give us a baseline projection of flops/s, which we modify with improvements from
    hardware specialization. The growth rate in flops/s defines the amortization period for the GPU, which then tells us
    the total FLOPs produced over the course of the GPU's lifetime. Then the cost of the GPU tells us FLOPs/$.
    """
    transistors_per_core_limit_samples = transistors_per_core_limit.sample(samples)
    process_size_limit_samples = process_size_limit.sample(samples)
    hardware_specialization_samples = hardware_specialization.sample(samples)

    # We use Marius's estimate to get a baseline projection, as a list of rollouts
    log_flops_per_second: List[List[float]] = gpu_efficiency.baseline_flops_per_second(
        samples, process_size_limit_samples, transistors_per_core_limit_samples
    )

    log_flops_per_dollar = []
    for rollout_idx, rollout in enumerate(log_flops_per_second):
        log_flops_per_dollar.append([])
        for year_offset in YEAR_OFFSETS:
            # Then, we modify the baseline with hardware_specialization
            rollout[year_offset] += (year_offset + 1) * hardware_specialization_samples[rollout_idx]

            # Use a reasonable, >10% growth rate for the first year
            growth_rate = 10**(rollout[year_offset] - rollout[year_offset - 1]) - 1 if year_offset else 0.5

            amortization_years = 1.2 / (growth_rate + 0.1)
            amortization_seconds = 365 * 24 * 60 * 60 * amortization_years
            log_flops_per_dollar[rollout_idx].append(rollout[year_offset] +
                                                     np.log10(amortization_seconds / gpu_dollar_cost))
    return np.stack(log_flops_per_dollar)


def algorithmic_improvements(
    # (24.6, 215.18) in percentages, converted to OOMs
    algo_growth_rate: DistributionCI = DistributionCI('normal', 95, np.log10(1.246), np.log10(315.18)).change_width(),
    transfer_multiplier: DistributionCI = DistributionCI('lognormal', 70, 0.4, 1.1).change_width(),
    algo_limit: DistributionCI = DistributionCI('lognormal', 70, 2, 10).change_width(),  # expressed in OOMs
    samples: int = NUM_SAMPLES,
):
    """
    Three components:
    - Base growth rate, from the "Algorithmic Progress in Computer Vision" paper
    - Domain transfer multiplier: how much we should modify the rate to account for algorithmic progress being a
    different domain.
    - Limit: at what multiplier does algorithmic growth stop?

    Growth slows as we approach the limit. Distribution values represent the quantity you should multiply physical
    compute by to get effective compute
    """
    transfer_multiplier_samples = transfer_multiplier.sample(samples)
    # Algorithmic regress is possible (due to regulation, knowledge loss, eg), which is represented by a growth_oom < 0
    algo_growth_samples = algo_growth_rate.sample(samples) * transfer_multiplier_samples
    limit_samples = algo_limit.sample(samples)

    algorithmic_improvement = []
    for rollout in range(samples):
        algorithmic_improvement.append([])
        for year_offset in YEAR_OFFSETS:
            current_improvement = (1 + year_offset) * algo_growth_samples[rollout]
            # Ensure the input to log isn't literally zero, which it can be sometimes, due to FP approximations when
            # the `value` is sufficently small compared to the `limit`
            constrained = max(1e-10, constrain(value=current_improvement, limit=limit_samples[rollout]))
            algorithmic_improvement[rollout].append(np.log10(constrained))

    return np.stack(algorithmic_improvement)


def tai_requirements(
    samples: int = NUM_SAMPLES,
    slowdown: DistributionCI = DistributionCI('lognormal', 70, 9.84, 290).change_width(),
    k_performance: DistributionCI = DistributionCI('lognormal', 70, 3129, 141714).change_width(),
    update_on_no_tai: bool = True,
) -> Union[tuple[Distribution, Distribution], Distribution]:
    """
    User specifies:
    - slowdown: the degree to which the human judge will update slower than the ideal predictor.
    - k_performance: the length of the transformative task
    """
    slowdown_samples = slowdown.sample(samples)
    k_performance_samples = k_performance.sample(samples)

    log_flops_needed_sample = []
    for i in range(len(slowdown_samples)):
        c = computation_for_k_performance(k_performance_samples[i], slowdown_samples[i])
        log_flops_needed_sample.append(c)

    log_flops_needed_sample = np.array(log_flops_needed_sample)

    if not update_on_no_tai:
        return log_flops_needed_sample

    compute_mu, compute_std = norm.fit(log_flops_needed_sample)

    training_runs = [(np.log10(1e17), 8), (np.log10(1e23), 2), (np.log10(1e25), 1)]

    basic_model = pm.Model()
    with basic_model:
        # compute requirements at which it takes 1 year to produce AGI in expectation
        compute_req_oom = pm.Normal("compute_req_oom", mu=compute_mu, sigma=compute_std)
        decay_exponent = 0.3

        i = 1
        for pair in training_runs:
            (run_size, duration) = pair
            poisson_mean = 10 ** (decay_exponent * (run_size - compute_req_oom))
            agi_arrived = pm.Poisson("agi_arrived_" + str(i), mu=poisson_mean * duration, observed=0)
            i += 1

    with basic_model:
        idata = pm.sample(draws=samples, tune=2000, cores=1, target_accept=0.99, progressbar=False)

    return log_flops_needed_sample, idata.posterior["compute_req_oom"].values[0]


def sample_timeline(
    samples: int = NUM_SAMPLES,
):
    compute_available = spending(samples=samples) + flops_per_dollar(samples=samples) + algorithmic_improvements(samples=samples)
    arrivals = compute_available.T > tai_requirements(samples=samples)[1]

    return np.sum(arrivals, axis=1) / samples


def get_default_params(timeline_func: Callable[..., Timeline]):
    param_dict = {}
    for param_name, param_val in inspect.signature(timeline_func).parameters.items():
        if param_name == 'samples':
            continue
        if isinstance(distribution_ci := param_val.default, DistributionCI):
            param_dict[param_name] = distribution_ci.params()
        else:
            param_dict[param_name] = param_val.default
    return param_dict