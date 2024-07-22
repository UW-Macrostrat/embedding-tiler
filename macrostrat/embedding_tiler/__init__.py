"""
Simple tile server that proxies a Macrostrat layer
"""
import os, logging
from dotenv import load_dotenv

# Sometimes we use environment variables to set
# module load paths etc.
load_dotenv()

# Now load the rest of the app
from fastapi import FastAPI, Request, Response, Query
from httpx import AsyncClient
from fastapi.middleware.cors import CORSMiddleware
from macrostrat.utils import setup_stderr_logs, get_logger
from macrostrat.utils.timer import Timer
from asyncio import sleep, get_running_loop

setup_stderr_logs("embedding_tiler", level=logging.INFO)

log = get_logger("embedding_tiler")

from .tile_processor import process_vector_tile_async

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

base_url = os.environ.get("MACROSTRAT_TILE_LAYER")
if "{z}" not in base_url:
    base_url += "/{z}/{x}/{y}"


# app.on_event("startup")(setup_model)

class ClientDisconnected(Exception):
    pass


async def check_client_disconnected(request: Request):
    if await request.is_disconnected():
        raise ClientDisconnected("Client disconnected")


@app.get("/search/{term}/tiles/{z}/{x}/{y}")
async def get_tile(request: Request, term: str, z: int, x: int, y: int, model: str = Query("BAAI/bge-base-en-v1.5")):
    tile_url = base_url.format(z=z, x=x, y=y)
    event_loop = get_running_loop()
    # Check for client disconnection:
    await check_client_disconnected(request)

    timer = Timer()
    with timer.context():
        async with AsyncClient(timeout=30) as client:
            log.info("Fetching tile x: %s, y: %s, z: %s", x, y, z)
            response = await client.get(tile_url)
            timer.add_step("fetch tile")

        await check_client_disconnected(request)

        res = await process_vector_tile_async(event_loop, response.content, term, model, timer)
    log_timings(timer)
    return Response(content=res, media_type="application/x-protobuf",
                    headers={"Server-Timing": timer.server_timings()})


def log_timings(timer: Timer):
    _timings = []
    for timing in timer.timings[1:]:
        _timings.append(f"{timing.name}: {timing.delta:.2f}")
    log.info("Timings: %s", ", ".join(_timings))


@app.exception_handler(ClientDisconnected)
def client_disconnected_handler(request: Request, exc: ClientDisconnected):
    log.info("Client disconnected")
    return Response(content="Client disconnected", status_code=499)


@app.get("/")
async def root():
    return {"base-url": base_url}
