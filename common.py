import sys

import numpy as np
import numpy.typing as npt
from typing import Literal, Optional
from scipy.stats import norm
import math

START_YEAR = 2023
END_YEAR = 2101
YEARS = list(range(START_YEAR, END_YEAR))
YEAR_OFFSETS = list(range(END_YEAR - START_YEAR))
NUM_SAMPLES = 20_000

# Types
Year = int
DistributionName = Literal['normal', 'lognormal']
Distribution = npt.NDArray[np.float64]  # shape: (num_samples,)
Rollout = npt.NDArray[np.float64]  # shape: (years,)
# rows are rollouts, columns are years, shape: (num_samples, years)
Timeline = npt.NDArray[np.float64]


class DistributionCI:
    def __init__(self, distribution: DistributionName, interval_width: float, interval_min: float, interval_max: float):
        assert interval_min <= interval_max, 'interval_min must be less than or equal to interval_max'
        assert distribution in ('normal', 'lognormal', 'delta'), 'Unsupported distribution'

        self.distribution = distribution
        self.interval_width = interval_width
        self.interval_min = interval_min
        self.interval_max = interval_max

        if distribution == 'lognormal' and interval_min == 0.0 and isinstance(interval_min, float):
            self.interval_min = sys.float_info.epsilon

    def mu(self):
        match self.distribution:
            case 'normal':
                return (self.interval_min + self.interval_max) / 2
            case 'lognormal':
                return (np.log(self.interval_min) + np.log(self.interval_max)) / 2
            case 'delta':
                return self.interval_min

    def sigma(self):
        match self.distribution:
            case 'normal':
                return (self.mu() - self.interval_min) / norm.interval(self.interval_width / 100)[1]
            case 'lognormal':
                return (self.mu() - np.log(self.interval_min)) / norm.interval(self.interval_width / 100)[1]
            case 'delta':
                return 0

    def sample(self, samples: int) -> npt.NDArray[np.float64]:
        if self.distribution != 'delta':
            assert 0 < self.interval_width < 100, 'interval_width must be a positive number representing a confidence interval'

        match self.distribution:
            case 'normal':
                return np.random.normal(self.mu(), self.sigma(), samples)
            case 'lognormal':
                return np.random.lognormal(self.mu(), self.sigma(), samples)
            case 'delta':
                return np.full(samples, self.interval_min)

    def params(self) -> dict:
        return {
            'distribution': self.distribution,
            'interval_width': self.interval_width,
            'interval_min': self.interval_min,
            'interval_max': self.interval_max,
        }

    def change_width(self, new_width: float = 80) -> 'DistributionCI':
        margin = norm.ppf((1 + new_width / 100) / 2) * self.sigma()
        match self.distribution:
            case 'normal':
                return DistributionCI(self.distribution, new_width, self.mu() - margin, self.mu() + margin)
            case 'lognormal':
                return DistributionCI(self.distribution, new_width,
                                      np.exp(self.mu() - margin), np.exp(self.mu() + margin))


def constrain(*, value: float, limit: float) -> float:
    """
    Implements `limit * (1 - exp(-value / limit))`, but a) takes inputs are in log space and returns values in linear
    space, and b) includes a couple kludges to account for floating point issues.
    """
    if math.isclose(limit, 0):
        return 0

    # Prevent overflow
    max_exponent = 300
    value = min(value, max_exponent)
    limit = min(limit, max_exponent)

    exp_res = np.exp(-10**(float(value - limit)))

    # I assume there's a better way to do this, but the goal is to avoid cases where `limit` is so much larger than
    # `value` that exp rounds to 1.0 (which is why we can use == rather than math.isclose)
    return 10**limit * (1 - exp_res) if exp_res != 1.0 else 10**value


def resample_between(samples: npt.NDArray[np.float64], min: Optional[float] = None, max: Optional[float] = None):
    assert min is not None or max is not None
    condition = np.full(samples.shape, True)
    if min:
        condition = condition & (samples > min)
    if max:
        condition = condition & (samples < max)

    matching_samples = samples[condition]
    samples_needed = len(samples) - len(matching_samples)
    return np.concatenate([matching_samples, np.array(np.random.choice(matching_samples, size=samples_needed))])
