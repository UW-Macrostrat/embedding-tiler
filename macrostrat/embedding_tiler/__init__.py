"""
Simple tile server that proxies a Macrostrat layer
"""
import os, logging
from dotenv import load_dotenv

# Sometimes we use environment variables to set
# module load paths etc.
load_dotenv()

# Now load the rest of the app
from fastapi import FastAPI, Request, Response
from httpx import AsyncClient
from fastapi.middleware.cors import CORSMiddleware
from macrostrat.utils import setup_stderr_logs, get_logger
from asyncio import sleep

setup_stderr_logs("embedding_tiler", level=logging.INFO)

log = get_logger("embedding_tiler")

from .tile_processor import process_vector_tile

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


@app.get("/tiles/{z}/{x}/{y}")
async def get_tile(request: Request, z: int, x: int, y: int):
    tile_url = base_url.format(z=z, x=x, y=y)
    log.info("Fetching tile x: %s, y: %s, z: %s", x, y, z)
    # Wait a tiny bit to start, in case we're zooming
    await sleep(0.1)
    # Check for client disconnection:
    if request.is_disconnected():
        return Response(content="Client disconnected", status_code=499)

    async with AsyncClient(timeout=30) as client:
        response = await client.get(tile_url)
        if request.is_disconnected():
            return Response(content="Client disconnected", status_code=499)

        res = process_vector_tile(response.content)
        return Response(content=res, media_type="application/x-protobuf")


@app.get("/")
async def root():
    return {"base-url": base_url}
