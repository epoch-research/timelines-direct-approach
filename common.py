import numpy as np
import numpy.typing as npt
from typing import Literal, Optional
from dataclasses import dataclass
from scipy.stats import norm
import math
from datetime import datetime

START_YEAR = 2023
END_YEAR = 2100
YEARS = list(range(START_YEAR, END_YEAR))
YEAR_OFFSETS = list(range(END_YEAR - START_YEAR))
NUM_SAMPLES = 10_000

# Types
Year = int
DistributionName = Literal['normal', 'lognormal']
Distribution = npt.NDArray[np.float64]  # shape: (num_samples,)
Rollout = npt.NDArray[np.float64]  # shape: (years,)
# rows are rollouts, columns are years, shape: (num_samples, years)
Timeline = npt.NDArray[np.float64]


@dataclass
class DistributionCI:
    distribution: DistributionName
    interval_width: float
    interval_min: float
    interval_max: float

    def sample(self, samples: int) -> npt.NDArray[np.float64]:
        assert 0 < self.interval_width < 100, 'interval_width must be a positive number representing a confidence interval'
        match self.distribution:
            case 'normal':
                mu = (self.interval_min + self.interval_max) / 2
                sigma = (mu - self.interval_min) / norm.interval(self.interval_width / 100)[1]
                return np.random.normal(mu, sigma, samples)
            case 'lognormal':
                mu = (np.log(self.interval_min) + np.log(self.interval_max)) / 2
                sigma = (mu - np.log(self.interval_min)) / norm.interval(self.interval_width / 100)[1]
                return np.random.lognormal(mu, sigma, samples)
            case _:
                raise ValueError(f"Unsupported distribution: {self.distribution}")

    def params(self) -> dict:
        return {
            'distribution': self.distribution,
            'interval_width': self.interval_width,
            'interval_min': self.interval_min,
            'interval_max': self.interval_max,
        }


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
