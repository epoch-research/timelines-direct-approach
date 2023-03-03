"""
Credit for large swathes of this goes to Marius Hobbhahn. Code below loads a model fitted to historical GPU trends.
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.regression.linear_model import RegressionResults
from datetime import datetime
import common
import itertools


def load_ndarray(name):
    return np.loadtxt(DATA_DIR + name + '.nparr.gz')


DATA_DIR = 'data/'
DATA_FILE_LOCATION = DATA_DIR + 'gpu_trends_factors_clean.csv'
GPU_TRENDS_CLEAN = pd.read_csv(DATA_FILE_LOCATION)
GPU_TRENDS_TIME_LOG10 = pd.DataFrame({
    "timestamp":GPU_TRENDS_CLEAN["timestamp"],
    "FP32_flops_log10":np.log10(GPU_TRENDS_CLEAN["FP32_flops"]),
    "FP32_flops_per_dollar_log10":np.log10(GPU_TRENDS_CLEAN["FP32_flops"]/GPU_TRENDS_CLEAN["real_price_2020"]),
    "transistors_in_millions_log10":np.log10(GPU_TRENDS_CLEAN["transistors_in_millions"]),
    "process_size_in_nm_log10":np.log10(GPU_TRENDS_CLEAN["process_size_in_nm"]),
    "GPU_clock_in_MHz_log10":np.log10(GPU_TRENDS_CLEAN["GPU_clock_in_MHz"]),
    "num_cores_log10":np.log10(GPU_TRENDS_CLEAN["num_cores"]),
    "memory_size_in_MB_log10":np.log10(GPU_TRENDS_CLEAN["memory_size_in_MB"]),
    "memory_clock_in_MHz_log10":np.log10(GPU_TRENDS_CLEAN["memory_clock_in_MHz"]),
    "bandwidth_in_GBs_log10":np.log10(GPU_TRENDS_CLEAN["bandwidth_in_GBs"]),
})
X2_TIME = pd.DataFrame({
    "timestamp":np.linspace(GPU_TRENDS_TIME_LOG10["timestamp"].min()-1e8,
                            GPU_TRENDS_TIME_LOG10["timestamp"].max()+1e9,
                            1000+1)
})
TIME_PROCESS_SIZE_PRED_X2_Q005 = load_ndarray('time_process_size_pred_x2_q005')
TIME_NUM_TRANSISTORS_RESULTS = RegressionResults.load(DATA_DIR + 'time_num_transistors_results.pkl')
MVP_results = RegressionResults.load(DATA_DIR + 'MVP_results.pkl')


def load_time_num_cores_perc_pred():
    arr_names = ['time_num_cores_pred_x2_q08', 'time_num_cores_pred_x2_q085',
                 'time_num_cores_pred_x2_q09', 'time_num_cores_pred_x2_q095']
    return [load_ndarray(arr_name) for arr_name in arr_names]


def limit_process_size_constraint(time_pred, limit_in_nm):
    return np.maximum(time_pred, np.log10(limit_in_nm))


def time_from_limit(time, process_size_pred, limit_in_nm):
    return time[process_size_pred == np.log10(limit_in_nm)].min().values


def ratio_transistors_cores(time_transistors_pred, time_cores_pred):
    return 10**time_transistors_pred / 10**time_cores_pred


def cores_from_transistors_and_ratio(time_transistors, ratio):
    return np.log10(10**time_transistors / ratio)


def get_year_start_indices(times) -> list[int]:
    """
    Given series of timestamps of the kind used for `X2_TIME`, return the indices which correspond to the first ts of
    a new year. Only include years >= to start_year
    """
    indices = []
    current_year = None
    for i, ts in enumerate(times.values):
        dt = datetime.fromtimestamp(ts[0])
        if dt.year >= common.START_YEAR and dt.year != current_year:
            current_year = dt.year
            indices.append(i)
    return indices


def predict_FLOPs_limit(
    limit_process_size,
    limit_transistors_per_core,
    time_span,
    time_cores_pred,
    num_transistors_pred_model,
    time_process_size_pred,
    MVP_model
):
    # compute limits from process_size_pred
    time_process_size_pred_limit = limit_process_size_constraint(time_process_size_pred, limit_process_size)
    timepoint_limit = time_from_limit(time_span, time_process_size_pred_limit, limit_process_size)

    # transfer limits from process size to num transistors
    num_transistors_pred_limit = num_transistors_pred_model.predict({"timestamp": [timepoint_limit]}).values
    time_num_transistors_pred = num_transistors_pred_model.predict({"timestamp": time_span})
    time_num_transistors_pred_limit = np.minimum(time_num_transistors_pred.values, num_transistors_pred_limit)

    # update cores prediction
    ratio_limit = ratio_transistors_cores(time_num_transistors_pred_limit, time_cores_pred)
    ratio_limit_capped = np.maximum(ratio_limit, limit_transistors_per_core)
    # translate this back to num_cores
    time_cores_pred_capped = cores_from_transistors_and_ratio(time_num_transistors_pred_limit, ratio_limit_capped)
    # compute timepoint where limit is hit
    timepoint_ratio_limit = time_span[time_cores_pred_capped == time_cores_pred_capped.max()].min().values

    # return FLOPs prediction
    process_size_cores_pred_dict = {
        "process_size_in_nm_log10": time_process_size_pred_limit,
        "num_cores_log10": time_cores_pred_capped
    }
    MVP_prediction_capped = MVP_model.predict(process_size_cores_pred_dict)
    return (MVP_prediction_capped, timepoint_limit, timepoint_ratio_limit)


def baseline_flops_per_second(
    num_samples: int,
    lognorm_process_size_limit_samples: npt.NDArray[np.float64],
    lognorm_transistors_per_core_limit_samples: npt.NDArray[np.float64],
) -> list[list[float]]:
    """ Return list of log10 rollouts in flops per second from start_year to end_year """
    time_num_cores_perc_pred = load_time_num_cores_perc_pred()
    year_start_indices = get_year_start_indices(X2_TIME)

    max_log_flops_per_second_rollouts = []
    for i in range(num_samples):
        # draw limits
        limit_transistors_per_core_sample = lognorm_transistors_per_core_limit_samples[i]
        limit_process_size_sample = lognorm_process_size_limit_samples[i]
        # draw percentile from 0.8, 0.85, 0.9 and 0.95
        j = np.random.randint(4, size=1)[0]
        time_num_cores_pred_x2_q = time_num_cores_perc_pred[j]
        pred, pred_limit, pred_ratio_limit = predict_FLOPs_limit(
          limit_process_size=limit_process_size_sample,
          limit_transistors_per_core=limit_transistors_per_core_sample,
          time_span=X2_TIME,
          time_cores_pred=time_num_cores_pred_x2_q,
          num_transistors_pred_model=TIME_NUM_TRANSISTORS_RESULTS,
          time_process_size_pred=TIME_PROCESS_SIZE_PRED_X2_Q005,
          MVP_model=MVP_results
        )
        max_log_flops_per_second_rollouts.append(list(pred.values[year_start_indices]))

    # Assume a baseline in which the last year of the projection is the best we can do until the timeline model ends
    return [rollout + list(itertools.repeat(rollout[-1], (len(common.YEARS) - len(rollout))))
            for rollout in max_log_flops_per_second_rollouts]
