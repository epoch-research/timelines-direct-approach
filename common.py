import sys

import numpy as np
import numpy.typing as npt
from typing import Literal, Optional
from scipy.stats import norm
import math

START_YEAR = 2023
END_YEAR = 2100
YEARS = list(range(START_YEAR, END_YEAR))
YEAR_OFFSETS = list(range(END_YEAR - START_YEAR))
NUM_SAMPLES = 6_000

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
        assert distribution in ('normal', 'lognormal'), 'Unsupported distribution'

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

    def sigma(self):
        match self.distribution:
            case 'normal':
                return (self.mu() - self.interval_min) / norm.interval(self.interval_width / 100)[1]
            case 'lognormal':
                return (self.mu() - np.log(self.interval_min)) / norm.interval(self.interval_width / 100)[1]

    def sample(self, samples: int) -> npt.NDArray[np.float64]:
        assert 0 < self.interval_width < 100, 'interval_width must be a positive number representing a confidence interval'
        match self.distribution:
            case 'normal':
                return np.random.normal(self.mu(), self.sigma(), samples)
            case 'lognormal':
                return np.random.lognormal(self.mu(), self.sigma(), samples)

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
    if math.isclose(limit, 0):
        return 0
    return limit * (1 - np.exp(-value / limit))


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
