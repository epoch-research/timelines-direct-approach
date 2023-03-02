import numpy as np
import numpy.typing as npt
import random
from typing import Literal
from dataclasses import dataclass
from scipy.stats import lognorm, norm
import math

# Just for development
np.random.seed(0)
random.seed(0)

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


def constrain(*, value: float, limit: float) -> float:
    if math.isclose(limit, 0):
        return 0
    return limit * (1 - np.exp(-value / limit))
