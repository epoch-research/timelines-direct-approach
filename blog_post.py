from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import numpy as np
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import io
from plots import plot_timeline, plot_tai_requirements, plot_tai_timeline, plot_tai_timeline_density
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from queue import Empty

import timeline
import common
from typing import TypedDict
import matplotlib

matplotlib.use('AGG')

TAI_REQUIREMENTS_PLOT_PARAMS = {'x_lab': 'FLOPs required'}
SPENDING_PLOT_PARAMS = {'y_lab': 'log10(Largest Training Run ($))'}
ALGORITHMIC_PROGRESS_PLOT_PARAMS = {'y_lab': 'log10(Algorithmic progress multiplier)'}
FLOPS_PER_DOLLAR_PLOT_PARAMS = {'y_lab': 'log10(FLOPs per $)'}
PHYSICAL_FLOPS_PLOT_PARAMS = {'y_lab': 'log10(Physical FLOPs)'}
EFFECTIVE_FLOPS_PLOT_PARAMS = {'y_lab': 'log10(Effective FLOPs)'}

DISTRIBUTION_CI_PARAM_KEYS = {'distribution', 'interval_width', 'interval_min', 'interval_max'}

DEFAULT_PARAMS = {
    'samples': common.NUM_SAMPLES,
    'spending': timeline.get_default_params(timeline.spending),
    'flops_per_dollar': timeline.get_default_params(timeline.flops_per_dollar),
    'algorithmic_improvements': timeline.get_default_params(timeline.algorithmic_improvements),
    'tai_requirements': timeline.get_default_params(timeline.tai_requirements),
}


# set default logging level
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

pool = ProcessPoolExecutor()


origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:4000",
    "https://epoch-backend-test.web.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DistributionCIParams(TypedDict):
    distribution: str
    interval_width: float
    interval_min: float
    interval_max: float


class SpendingParams(TypedDict):
    invest_growth_rate: DistributionCIParams
    gwp_growth_rate: DistributionCIParams
    max_gwp_pct: DistributionCIParams
    starting_gwp: float
    starting_max_spend: float


class FlopsPerDollarParams(TypedDict):
    transistors_per_core_limit: DistributionCIParams
    process_size_limit: DistributionCIParams
    hardware_specialization: DistributionCIParams
    gpu_dollar_cost: int


class AlgorithmicImprovementsParams(TypedDict):
    growth_rate: DistributionCIParams
    transfer_multiplier: DistributionCIParams
    limit: DistributionCIParams


class TAIRequirementsParams(TypedDict):
    slowdown: DistributionCIParams
    k_performance: DistributionCIParams


# TODO: think about constraints on values here
class TimelineParams(BaseModel):
    samples: int
    spending: SpendingParams
    flops_per_dollar: FlopsPerDollarParams
    algorithmic_improvements: AlgorithmicImprovementsParams
    tai_requirements: TAIRequirementsParams


@app.on_event("startup")
async def app_startup():
    timeline_params = make_json_params_callable(DEFAULT_PARAMS)

    tai_requirements = timeline.tai_requirements(**timeline_params['tai_requirements'])
    with open('static/tai_requirements.png', 'wb') as f:
        plot_tai_requirements(tai_requirements, **TAI_REQUIREMENTS_PLOT_PARAMS).savefig(f)

    algorithmic_progress_timeline = timeline.algorithmic_improvements(**timeline_params['algorithmic_improvements'])
    with open('static/algorithmic_progress.png', 'wb') as f:
        plot_timeline(algorithmic_progress_timeline, **ALGORITHMIC_PROGRESS_PLOT_PARAMS).savefig(f)

    investment_timeline = timeline.spending(**timeline_params['spending'])
    with open('static/spending.png', 'wb') as f:
        plot_timeline(investment_timeline, **SPENDING_PLOT_PARAMS).savefig(f)

    flops_per_dollar_timeline = timeline.flops_per_dollar(**timeline_params['flops_per_dollar'])
    physical_flops_timeline = flops_per_dollar_timeline + investment_timeline
    effective_compute_timeline = physical_flops_timeline + algorithmic_progress_timeline

    with open('static/flops_per_dollar.png', 'wb') as f:
        plot_timeline(flops_per_dollar_timeline, **FLOPS_PER_DOLLAR_PLOT_PARAMS).savefig(f)

    with open('static/physical_flops.png', 'wb') as f:
        plot_timeline(physical_flops_timeline, **PHYSICAL_FLOPS_PLOT_PARAMS).savefig(f)

    with open('static/effective_flops.png', 'wb') as f:
        plot_timeline(effective_compute_timeline, **EFFECTIVE_FLOPS_PLOT_PARAMS).savefig(f)

    arrivals = effective_compute_timeline.T > tai_requirements
    tai_timeline = np.sum(arrivals, axis=1) / timeline_params['samples']

    with open('static/tai_timeline.png', 'wb') as f:
        plot_tai_timeline(tai_timeline).savefig(f)

    with open('static/tai_timeline_density.png', 'wb') as f:
        plot_tai_timeline_density(arrivals).savefig(f)

    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted")


@app.get("/timeline-defaults")
def get_timeline_defaults():
    content = {"defaults": DEFAULT_PARAMS}
    return JSONResponse(content=content)


def make_json_params_callable(json_params):
    timeline_params = {'samples': json_params['samples']}
    for timeline_function in ['spending', 'flops_per_dollar', 'algorithmic_improvements', 'tai_requirements']:
        timeline_params[timeline_function] = {'samples': json_params['samples']}
        for k in json_params[timeline_function]:
            v = json_params[timeline_function][k]
            if isinstance(v, dict) and set(v.keys()) == DISTRIBUTION_CI_PARAM_KEYS:
                timeline_params[timeline_function][k] = common.DistributionCI(**v)
    return timeline_params


@app.websocket("/generate-timeline")
async def generate_timeline(websocket: WebSocket):
    await websocket.accept()

    # TODO: validate params
    timeline_params = make_json_params_callable(json.loads(await websocket.receive_text()))
    logger.info(timeline_params)

    loop = asyncio.get_event_loop()
    manager = mp.Manager()
    queue = manager.Queue()

    generate_timeline_process = loop.run_in_executor(pool, generate_timeline_plots, timeline_params, queue)
    while True:
        await asyncio.sleep(0.5)
        try:
            plot_fig = queue.get(block=False)
            await websocket.send_bytes(plot_fig)
        except Empty:
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
            plot_fig = queue.get(block=False)
            await websocket.send_bytes(plot_fig)
        except Empty:
            break

    await websocket.close()


def generate_timeline_plots(timeline_params, queue):
    tai_requirements = timeline.tai_requirements(**timeline_params['tai_requirements'])
    with io.BytesIO() as f:
        plot_tai_requirements(tai_requirements, **TAI_REQUIREMENTS_PLOT_PARAMS).savefig(f)
        queue.put(f.getvalue())

    algorithmic_progress_timeline = timeline.algorithmic_improvements(**timeline_params['algorithmic_improvements'])
    with io.BytesIO() as f:
        plot_timeline(algorithmic_progress_timeline, **ALGORITHMIC_PROGRESS_PLOT_PARAMS).savefig(f)
        queue.put(f.getvalue())

    investment_timeline = timeline.spending(**timeline_params['spending'])
    with io.BytesIO() as f:
        plot_timeline(investment_timeline, **SPENDING_PLOT_PARAMS).savefig(f)
        queue.put(f.getvalue())

    flops_per_dollar_timeline = timeline.flops_per_dollar(**timeline_params['flops_per_dollar'])
    physical_flops_timeline = flops_per_dollar_timeline + investment_timeline
    effective_compute_timeline = physical_flops_timeline + algorithmic_progress_timeline

    with io.BytesIO() as f:
        plot_timeline(flops_per_dollar_timeline, **FLOPS_PER_DOLLAR_PLOT_PARAMS).savefig(f)
        queue.put(f.getvalue())

    with io.BytesIO() as f:
        plot_timeline(physical_flops_timeline, **PHYSICAL_FLOPS_PLOT_PARAMS).savefig(f)
        queue.put(f.getvalue())

    with io.BytesIO() as f:
        plot_timeline(effective_compute_timeline, **EFFECTIVE_FLOPS_PLOT_PARAMS).savefig(f)
        queue.put(f.getvalue())

    arrivals = effective_compute_timeline.T > tai_requirements
    tai_timeline = np.sum(arrivals, axis=1) / timeline_params['samples']

    with io.BytesIO() as f:
        plot_tai_timeline(tai_timeline).savefig(f)
        queue.put(f.getvalue())

    with io.BytesIO() as f:
        plot_tai_timeline_density(arrivals).savefig(f)
        queue.put(f.getvalue())
