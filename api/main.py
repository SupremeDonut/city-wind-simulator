from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from routes import presets, predict, simulate, results, map_texture

app = FastAPI(title="Wind Simulation API")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(presets.router)
app.include_router(predict.router)
app.include_router(simulate.router)
app.include_router(results.router)
app.include_router(map_texture.router)
