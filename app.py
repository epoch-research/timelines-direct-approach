import asyncio
import io
import json
import logging
import multiprocessing as mp
import queue
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TypedDict, Literal
from typing import Dict, List, Union

import matplotlib
import seaborn
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator, conint, confloat, ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

import common
import timeline
from plots import plot_timeline, plot_tai_requirements, plot_tai_timeline, plot_tai_timeline_density
from plots import plot_tai_requirements_adjustment_decomposition


seaborn.set_theme()
matplotlib.use('AGG')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.titlesize'] = 11.5
np.random.seed(2024)

POOL = None

TAI_REQUIREMENTS_PLOT_PARAMS = {'x_lab': 'log(FLOP)', 'title': 'Distribution over effective FLOP before update on upper bound'}
TAI_REQUIREMENTS_ADJUSTMENT_DECOMPOSITION_PLOT_PARAMS = {'x_lab': 'log(FLOP)', 'title': 'Decomposition of the adjustment (PDF)'}
CUMULATIVE_TAI_REQUIREMENTS_ADJUSTMENT_DECOMPOSITION_PLOT_PARAMS = {'x_lab': 'log(FLOP)', 'cumulative': True, 'title': 'Decomposition of the adjustment (CDF)'}
ADJUSTED_TAI_REQUIREMENTS_PLOT_PARAMS = {'x_lab': 'log(FLOP)', 'title': 'Distribution over effective FLOP after update on upper bound'}
CUMULATIVE_TAI_REQUIREMENTS_PLOT_PARAMS = {'x_lab': 'log(FLOP)', 'cumulative': True,
                                           'title': 'Cumulative distribution over effective FLOP required for TAI after adjustment'}
SPENDING_PLOT_PARAMS = {'y_lab': 'Largest Training Run ($)'}
ALGORITHMIC_PROGRESS_PLOT_PARAMS = {'y_lab': 'Algorithmic progress multiplier'}
FLOPS_PER_DOLLAR_PLOT_PARAMS = {'y_lab': 'FLOP/$'}
PHYSICAL_FLOPS_PLOT_PARAMS = {'y_lab': 'Physical FLOP'}
EFFECTIVE_FLOPS_PLOT_PARAMS = {'y_lab': 'Effective FLOP'}
TAI_TIMELINE_PLOT_PARAMS = {'x_lab': 'Year', 'y_lab': 'P(TAI)', 'title': 'Cumulative probability of TAI arrival'}
TAI_TIMELINE_DENSITY_PLOT_PARAMS = {'x_lab': 'Year', 'y_lab': 'P(TAI)',  'title': 'Distribution over TAI arrival year'}

DISTRIBUTION_CI_PARAM_KEYS = {'distribution', 'interval_width', 'interval_min', 'interval_max'}

DEFAULT_PARAMS = {
    'samples': common.NUM_SAMPLES,
    'spending': timeline.get_default_params(timeline.spending),
    'flops_per_dollar': timeline.get_default_params(timeline.flops_per_dollar),
    'algorithmic_improvements': timeline.get_default_params(timeline.algorithmic_improvements),
    'tai_requirements': timeline.get_default_params(timeline.tai_requirements),
}

logging.basicConfig(
    handlers=[logging.FileHandler("timelines_backend.log"), logging.StreamHandler()],
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.getLogger('pymc').setLevel(logging.WARNING)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DistributionCIParams(TypedDict):
    distribution: Literal["normal", "lognormal"]
    interval_width: float
    interval_min: float
    interval_max: float


class SpendingParams(BaseModel):
    invest_growth_rate: DistributionCIParams
    gwp_growth_rate: DistributionCIParams
    max_gwp_pct: DistributionCIParams
    starting_gwp: float
    starting_max_spend: confloat(gt=0)

    @validator("invest_growth_rate", "gwp_growth_rate", pre=True)
    def check_lower_bound_gt_neg_one(cls, value):
        if value["interval_min"] <= -100:
            raise ValueError("The lower bound of the interval must be > -100")
        return value


class FlopsPerDollarParams(BaseModel):
    transistors_per_core_limit: DistributionCIParams
    process_size_limit: DistributionCIParams
    hardware_specialization: DistributionCIParams
    gpu_dollar_cost: conint(gt=0)


class AlgorithmicImprovementsParams(BaseModel):
    algo_growth_rate: DistributionCIParams
    transfer_multiplier: DistributionCIParams
    algo_limit: DistributionCIParams

    @validator("transfer_multiplier", pre=True)
    def check_transfer_lower_bound_gt_zero(cls, value):
        if value["interval_min"] <= 0:
            raise ValueError("The lower bound of the interval must be > 0")
        return value

    @validator("algo_limit", pre=True)
    def check_limit_lower_bound_ge_zero(cls, value):
        if value["interval_min"] < 0:
            raise ValueError("The lower bound of the interval must be >= 0")
        return value


class TAIRequirementsParams(BaseModel):
    slowdown: DistributionCIParams
    k_performance: DistributionCIParams
    upper_bound_weight: confloat(ge=0, le=1)

    @validator("slowdown", "k_performance", pre=True)
    def check_lower_bound_gt_zero(cls, value):
        if value["interval_min"] <= 0:
            raise ValueError("The lower bound of the interval must be > 0")
        return value


class TimelineParams(BaseModel):
    samples: conint(gt=0, le=5000, strict=True)
    spending: SpendingParams
    flops_per_dollar: FlopsPerDollarParams
    algorithmic_improvements: AlgorithmicImprovementsParams
    tai_requirements: TAIRequirementsParams

    @validator("spending", "flops_per_dollar", "algorithmic_improvements", "tai_requirements", pre=True)
    def check_min_less_than_max(cls, value):
        for k, v in value.items():
            if isinstance(v, dict) and "interval_min" in v and "interval_max" in v:
                if v["interval_min"] > v["interval_max"]:
                    raise ValueError(f"{k}: the lower bound for an interval must be <= the upper bound")
                if v["interval_min"] < 0 and v["distribution"] == "lognormal":
                    raise ValueError(f"{k}: the lower bound for a lognormal interval must be >= 0")
        return value


def show_pydantic_errors(errs):
    full_msg = "There were some problems with your input:"
    for err in errs:
        loc = " -> ".join(err['loc'])
        msg = err['msg']
        if ':' in msg:
            loc += ' -> ' + msg.split(':')[0]
            msg = msg.split(':')[1]
        full_msg += f"\n• {loc}: {msg}"
    return full_msg


@app.on_event("startup")
async def app_startup():
    global POOL
    POOL = ProcessPoolExecutor()

    q = queue.SimpleQueue()

    timeline_params = make_json_params_callable(DEFAULT_PARAMS)
    summary = generate_timeline_plots(timeline_params, q)

    with open('static/timeline-summary.json', 'w') as f:
        json.dump(summary, f)

    plot_names = ['tai_requirements', 'tai_requirements_adjustment_decomposition', 'cumulative_tai_requirements_adjustment_decomposition',
                  'adjusted_tai_requirements', 'cumulative_adjusted_tai_requirements',
                  'algorithmic_progress', 'spending', 'flops_per_dollar', 'physical_flops', 'effective_flops',
                  'tai_timeline', 'tai_timeline_density']

    for plot_name in plot_names:
        with open(f'static/{plot_name}.png', 'wb') as f:
            f.write(q.get())

    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted")


@app.on_event("shutdown")
def shutdown():
    POOL.shutdown(wait=True)


@app.get("/timeline-defaults")
def get_timeline_defaults():
    content = {"defaults": DEFAULT_PARAMS}
    # Nudge people a bit to reduce the load
    content["defaults"]["samples"] = 2000

    with open("static/timeline-summary.json") as f:
        content["timeline_summary"] = json.load(f)

    return JSONResponse(content=content)


def make_json_params_callable(json_params):
    timeline_params = {'samples': json_params['samples']}
    for timeline_function in ['spending', 'flops_per_dollar', 'algorithmic_improvements', 'tai_requirements']:
        timeline_params[timeline_function] = {'samples': json_params['samples']}
        for k in json_params[timeline_function]:
            v = json_params[timeline_function][k]
            if isinstance(v, dict) and set(v.keys()) == DISTRIBUTION_CI_PARAM_KEYS:
                timeline_params[timeline_function][k] = common.DistributionCI(**v)
            else:
                timeline_params[timeline_function][k] = v
    return timeline_params


@app.post("/validate-params")
async def validate_params(req: Request):
    params = await req.json()
    logger.info(f'Validating {type(params)} {params}')
    try:
        TimelineParams.parse_obj(params)
        logger.info('parsed params')
        return JSONResponse(content={"valid": True})
    except ValidationError as e:
        logger.exception('Error validating params')
        return JSONResponse(content={"valid": False, "error_message": show_pydantic_errors(e.errors())})


@app.websocket("/generate-timeline")
async def generate_timeline(websocket: WebSocket):
    global POOL

    await websocket.accept()
    start_time = time.time()

    json_params = json.loads(await websocket.receive_text())
    timeline_params = make_json_params_callable(json_params)
    logger.info(f'Calculating with {json_params}')

    loop = asyncio.get_event_loop()
    manager = mp.Manager()
    q = manager.Queue()

    try:
        generate_timeline_process = loop.run_in_executor(POOL, generate_timeline_plots, timeline_params, q)
    except BrokenProcessPool:
        logger.error('Process pool is broken. Restarting executor.')
        POOL.shutdown(wait=True)
        POOL = ProcessPoolExecutor()
        raise

    while True:
        await asyncio.sleep(0.1)
        try:
            await websocket.send_bytes(q.get(block=False))
        except queue.Empty:
            pass
        except WebSocketDisconnect:
            generate_timeline_process.cancel()
            break

        if generate_timeline_process.done():
            if e := generate_timeline_process.exception():
                raise e
            break

    # Clear the queue
    while True:
        try:
            await websocket.send_bytes(q.get(block=False))
        except queue.Empty:
            break

    # Send the timeline summary
    await websocket.send_json(generate_timeline_process.result())

    logger.info(f"Generated {timeline_params['samples']} sample timeline "
                f"in {round(time.time() - start_time, 1)} seconds")

    await websocket.close()


def put_plot(fig: matplotlib.figure.Figure, q: Union[queue.SimpleQueue, mp.Queue]):
    with io.BytesIO() as f:
        fig.savefig(f, dpi=200)
        q.put(f.getvalue())
    matplotlib.pyplot.close(fig)


def generate_timeline_plots(timeline_params, q: Union[queue.SimpleQueue, mp.Queue]) -> Dict[str, List[str]]:
    tai_requirements, upper_bound_samples, prior, upper_bound, posterior = timeline.tai_requirements(**timeline_params['tai_requirements'])
    put_plot(plot_tai_requirements(upper_bound_samples, **TAI_REQUIREMENTS_PLOT_PARAMS), q)
    put_plot(plot_tai_requirements_adjustment_decomposition(prior, upper_bound, posterior, **TAI_REQUIREMENTS_ADJUSTMENT_DECOMPOSITION_PLOT_PARAMS), q)
    put_plot(plot_tai_requirements_adjustment_decomposition(prior, upper_bound, posterior, **CUMULATIVE_TAI_REQUIREMENTS_ADJUSTMENT_DECOMPOSITION_PLOT_PARAMS), q)
    put_plot(plot_tai_requirements(tai_requirements, **ADJUSTED_TAI_REQUIREMENTS_PLOT_PARAMS), q)
    put_plot(plot_tai_requirements(tai_requirements, **CUMULATIVE_TAI_REQUIREMENTS_PLOT_PARAMS), q)

    algorithmic_progress_timeline = timeline.algorithmic_improvements(**timeline_params['algorithmic_improvements'])
    put_plot(plot_timeline(algorithmic_progress_timeline, **ALGORITHMIC_PROGRESS_PLOT_PARAMS), q)

    investment_timeline = timeline.spending(**timeline_params['spending'])
    put_plot(plot_timeline(investment_timeline, **SPENDING_PLOT_PARAMS), q)

    flops_per_dollar_timeline = timeline.flops_per_dollar(**timeline_params['flops_per_dollar'])
    physical_flops_timeline = flops_per_dollar_timeline + investment_timeline
    effective_compute_timeline = physical_flops_timeline + algorithmic_progress_timeline

    put_plot(plot_timeline(flops_per_dollar_timeline, **FLOPS_PER_DOLLAR_PLOT_PARAMS), q)
    put_plot(plot_timeline(physical_flops_timeline, **PHYSICAL_FLOPS_PLOT_PARAMS), q)
    put_plot(plot_timeline(effective_compute_timeline, **EFFECTIVE_FLOPS_PLOT_PARAMS), q)

    arrivals = effective_compute_timeline.T > tai_requirements
    tai_timeline = np.sum(arrivals, axis=1) / timeline_params['samples']
    median_arrival = quantile(tai_timeline, 0.5)

    put_plot(plot_tai_timeline(tai_timeline, median_arrival, **TAI_TIMELINE_PLOT_PARAMS), q)
    put_plot(plot_tai_timeline_density(arrivals, median_arrival, **TAI_TIMELINE_DENSITY_PLOT_PARAMS), q)

    return timeline_summary(tai_timeline)


def timeline_summary(tai_timeline: common.Timeline) -> Dict[str, List[str]]:
    quantiles = [quantile(tai_timeline, q) for q in [0.10, 0.5, 0.9]]
    return {
        "probabilities": [str(round(100 * tai_timeline[year - common.START_YEAR])) + "%" for year in [2030, 2050, 2100]],
        "quantiles": [str(q) if not np.isnan(q) else ">2100" for q in quantiles],
    }


def quantile(tai_timeline, q):
    for year_offset, cum_prob in enumerate(tai_timeline):
        if cum_prob >= q:
            return year_offset + common.START_YEAR
    return np.nan
