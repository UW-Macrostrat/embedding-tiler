"""
Simple tile server that proxies a Macrostrat layer
"""
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from httpx import AsyncClient
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

base_url = os.environ.get("MACROSTRAT_TILE_LAYER")
if "{z}" not in base_url:
    base_url += "/{z}/{x}/{y}"

@app.get("/tiles/{z}/{x}/{y}")
async def get_tile(request: Request, z: int, x: int, y: int):
    tile_url = base_url.format(z=z, x=x, y=y)
    async with AsyncClient(timeout=30) as client:
        response = await client.get(tile_url)
        return Response(content=response.content, media_type="application/x-protobuf")



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"base-url": base_url}
