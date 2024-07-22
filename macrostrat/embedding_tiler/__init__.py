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
from httpx import Client
from fastapi.middleware.cors import CORSMiddleware
from macrostrat.utils import setup_stderr_logs, get_logger

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
def get_tile(request: Request, z: int, x: int, y: int):
    tile_url = base_url.format(z=z, x=x, y=y)
    log.info("Fetching tile x: %s, y: %s, z: %s", x, y, z)
    with Client(timeout=30) as client:
        response = client.get(tile_url)
        res = process_vector_tile(response.content)
        return Response(content=res, media_type="application/x-protobuf")


@app.get("/")
async def root():
    return {"base-url": base_url}
