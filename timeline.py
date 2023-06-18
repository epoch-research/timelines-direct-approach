import inspect
from typing import List, Tuple, Callable, Union

import sympy as sp
import numpy as np
import numpy.typing as npt
from scipy.stats import norm, gaussian_kde, rv_continuous
from scipy.integrate import quad, cumulative_trapezoid

import gpu_efficiency
from common import DistributionCI, Timeline, Distribution, NUM_SAMPLES, YEAR_OFFSETS, constrain, resample_between, RNG, CURRENT_LARGEST_TRAINING_RUN
from k_performance import computation_for_k_performance

import matplotlib.pyplot as plt


class UninformativeTaiFlopPrior(rv_continuous):
    def __init__(self, current_largest_training_run, *args, **kwargs):
        super().__init__(a=current_largest_training_run, *args, **kwargs)

        x = sp.Symbol('x')

        # TODO Ask for the rationale behind this distribution
        cdf = sp.Piecewise(
            (0, x < current_largest_training_run),
            (1 - (current_largest_training_run/x)**(1/3), True),
        )

        pdf = sp.diff(cdf, x)

        self._cdf = sp.lambdify((x,), cdf)
        self._pdf = sp.lambdify((x,), pdf)


class TaiFlopPosterior(rv_continuous):
    r"""
    Update a prior on TAI requirements given an upper bound distribution, using this update rule:

        p^{\prime}(x) = s * \int_{x}^{\infty} p(x) \frac{\text{upper_bound}(u)}{\text{CDF}(u)} du,

    where

        s = \frac{1}{1 - CDF(min(support(upper_bound), support(p)))}.

    TODO: Add link to writeup
    """

    def __init__(self, prior: rv_continuous, upper_bound: rv_continuous, *args, **kwargs):
        super().__init__(a=max(prior.a, upper_bound.a), *args, **kwargs)

        self.prior = prior
        self.upper_bound = upper_bound

    def _pdf(self, x):
        # Computes the update rule in a sort of efficient way

        f = lambda u: self.upper_bound.pdf(u)/self.prior.cdf(u)

        result = self.prior.pdf(x) * self.reverse_cumulative_quad(f, x, self.a)
        result /= (1 - self.upper_bound.cdf(self.a))

        return result

    def _cdf(self, x):
        # Automatically integrating the PDF is a bit difficult. We do this manually ourselves.

        f = lambda u: self.upper_bound.pdf(u)/self.prior.cdf(u)

        result = [x > self.a] * ((self.upper_bound.cdf(x) - self.upper_bound.cdf(self.a)) + self.prior.cdf(x) * self.reverse_cumulative_quad(f, x, self.a))
        result /= (1 - self.upper_bound.cdf(self.a))

        return result

    def reverse_cumulative_quad(self, f: Callable, x: npt.NDArray, x0: float) -> npt.NDArray:
        assert np.all(np.diff(x) >= 0), 'x must be sorted'

        quads = []
        for i in range(len(x)):
            a = x[i]
            b = x[i+1] if i < len(x) - 1 else np.inf
            q = 0 if (a < x0) else quad(f, a, b)[0]
            quads.append(q)

        return np.cumsum(quads[::-1])[::-1]


class CombinationDistribution(rv_continuous):
    """Combination of two continuous distributions. The combined distribution is

        left_weight * dist_left + (1 - left_weight) * dist_right
    """

    def __init__(self, dist_left: rv_continuous, dist_right: rv_continuous, left_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist_left = dist_left
        self.dist_right = dist_right
        self.left_weight = left_weight

    def _pdf(self, x):
        p = self.left_weight * self.dist_left.pdf(x) + (1 - self.left_weight) * self.dist_right.pdf(x)
        return p

    def _cdf(self, x):
        c = self.left_weight * self.dist_left.cdf(x) + (1 - self.left_weight) * self.dist_right.cdf(x)
        return c


class GriddedDistribution(rv_continuous):
    """Generates a distribution by sampling a PDF in a grid and then interpolate it. This might help performance."""

    def __init__(self, dist: Union[rv_continuous, gaussian_kde], grid: npt.NDArray, *args, **kwargs):
        super().__init__(a=min(grid), b=max(grid), *args, **kwargs)

        self.grid = grid
        self.pdf_grid = dist.pdf(grid)

        if isinstance(dist, gaussian_kde):
            # gaussian_kdes don't have a .cdf() method
            self.cdf_grid = cumulative_trapezoid(self.pdf_grid, self.grid, initial=0)
        else:
            self.cdf_grid = dist.cdf(grid)

    def _pdf(self, x):
        return np.interp(x, self.grid, self.pdf_grid)

    def _cdf(self, x):
        return np.interp(x, self.grid, self.cdf_grid)


def spending(
    samples: int = NUM_SAMPLES,
    # from Ben's estimate of 0.1 to 0.3 OOMs per year in https://epochai.org/blog/trends-in-the-dollar-training-cost-of-machine-learning-systems#appendix-i-overall-best-guess-for-the-growth-rate-in-training-cost:~:text=my%20all%2Dthings%2Dconsidered%20view%20is%200.2%20OOMs/year%20(90%25%20CI%3A%200.1%20to%200.3%20OOMs/year
    # For ref: Epoch staff aggregate: DistributionCI('normal', 70, 1.1480341, 3.278781)
    invest_growth_rate: DistributionCI = DistributionCI('normal', 90, 25.9, 99.5).change_width(),
    # 116_639_700 in 2023 -> 237_830_800 in 2060 -> 1.9% per year
    gwp_growth_rate: DistributionCI = DistributionCI('normal', 80, 0.4, 4.75).change_width(),
    # Ben's estimate (ie 0.004% to 5%) (Matthew's estimate: DistributionCI('lognormal', 70, 0.0000083, 0.014047))
    max_gwp_pct: DistributionCI = DistributionCI('lognormal', 95, 0.004, 5).change_width(),
    starting_gwp: float = 1.17e14,
    # millions of dollars
    starting_max_spend: float = 60,
) -> Timeline:
    """
    We assume that the current maximum amount people are willing to spend on a training run is $100 million, which will
    grow at `invest_growth_rate` until we reach the `max_gwp_pct` of GWP, at which point it will grow at the rate of GWP
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
    hardware_specialization: DistributionCI = DistributionCI('lognormal', 80, 0.04, 0.25).change_width(),
    # expressed in OOMs
    hardware_specialiation_limit: float = np.log10(250),
    # Chosen to get to a starting FLOP/$ of ~4e17
    gpu_dollar_cost: int = 5_000,
) -> Timeline:
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
            # Then, we modify the baseline with hardware_specialization (up to a limit)
            hard_spec_gains = (year_offset + 1) * hardware_specialization_samples[rollout_idx]

            rollout[year_offset] += min(hard_spec_gains, hardware_specialiation_limit)

            # Use a reasonable, >10% growth rate for the first year
            growth_rate = 10**(rollout[year_offset] - rollout[year_offset - 1]) - 1 if year_offset else 0.5

            amortization_years = 1.2 / (growth_rate + 0.1)
            amortization_seconds = 365 * 24 * 60 * 60 * amortization_years
            log_flops_per_dollar[rollout_idx].append(rollout[year_offset] +
                                                     np.log10(amortization_seconds / gpu_dollar_cost))
    return np.stack(log_flops_per_dollar)


def algorithmic_improvements(
    # (24.6, 215.18) in percentages, converted to OOMs
    algo_growth_rate: DistributionCI = DistributionCI('normal', 80, 0.35, 0.75).change_width(),
    transfer_multiplier: DistributionCI = DistributionCI('lognormal', 70, 0.4, 1.1).change_width(),
    algo_limit: DistributionCI = DistributionCI('lognormal', 80, 2.25, 12.1).change_width(),  # expressed in OOMs
    samples: int = NUM_SAMPLES,
) -> Timeline:
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
    upper_bound_weight: float = 0.9,
) -> Tuple[Distribution, Distribution, rv_continuous, rv_continuous, rv_continuous, rv_continuous]:
    """
    User specifies:
    - slowdown: the degree to which the human judge will update slower than the ideal predictor.
    - k_performance: the length of the transformative task
    - prior_weight: how much weight to give to the prior vs the updated posterior
    """

    slowdown_samples = slowdown.sample(samples)
    k_performance_samples = k_performance.sample(samples)

    upper_bound_samples = []
    for i in range(len(slowdown_samples)):
        c = computation_for_k_performance(k_performance_samples[i], slowdown_samples[i])
        upper_bound_samples.append(c)

    upper_bound_samples = np.array(upper_bound_samples)

    # use the upper bound above to update an uninformative prior

    current_largest_training_run = 25

    approx_grid = np.linspace(1, 100, 500)
    approx_grid = np.sort(np.unique(approx_grid))

    prior       = UninformativeTaiFlopPrior(CURRENT_LARGEST_TRAINING_RUN)
    upper_bound = GriddedDistribution(gaussian_kde(upper_bound_samples), grid=approx_grid)
    posterior   = GriddedDistribution(TaiFlopPosterior(prior, upper_bound), grid=approx_grid)
    combination = CombinationDistribution(upper_bound, posterior, left_weight=upper_bound_weight)

    tai_requirements_samples = combination.rvs(size=samples)

    return tai_requirements_samples, upper_bound_samples, prior, upper_bound, posterior, combination


def sample_timeline(
    samples: int = NUM_SAMPLES,
):
    compute_available = spending(samples=samples) + flops_per_dollar(samples=samples) + algorithmic_improvements(samples=samples)
    arrivals = compute_available.T > tai_requirements(samples=samples)[0]

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
