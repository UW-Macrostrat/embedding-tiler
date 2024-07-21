"""
Simple tile server that proxies a Macrostrat layer
"""
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from httpx import AsyncClient

load_dotenv()

app = FastAPI()

base_url = os.environ.get("MACROSTRAT_TILE_LAYER")

@app.get("/tiles/{z}/{x}/{y}")
async def get_tile(request: Request, z: int, x: int, y: int):
    tile_url = base_url.format(z=z, x=x, y=y)
    async with AsyncClient() as client:
        response = await client.get(tile_url)
        return Response(content=response.content, headers=response.headers)
