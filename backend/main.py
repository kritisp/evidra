"""
EVIDRA — Evidence-Grounded Document Intelligence
FastAPI Application Entry Point
"""

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from routers import chat, documents, health, tools
from services.chroma_service import init_chroma


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize persistent ChromaDB vector store
    init_chroma()
    print("✅ EVIDRA Vector Database ready.")
    yield
    print("👋 EVIDRA Backend shutting down.")


app = FastAPI(
    title="EVIDRA — Evidence-Grounded Document Intelligence",
    description="Production-grade Document-Grounded RAG API with verifiable page-level citations and strict abstention.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core API endpoints
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat & Grounding"])
app.include_router(documents.router, prefix="/api", tags=["Documents & Ingestion"])
app.include_router(tools.router, prefix="/api", tags=["Smart Tools"])

# Static frontend delivery (prefers Vite production build in dist/ if available)
frontend_dist_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
frontend_raw_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
frontend_path = frontend_dist_path if os.path.exists(frontend_dist_path) else frontend_raw_path

if os.path.exists(frontend_path):
    assets_path = os.path.join(frontend_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/", response_class=FileResponse)
    async def serve_index():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{full_path:path}", response_class=FileResponse)
    async def serve_spa(full_path: str):
        file_path = os.path.join(frontend_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))
