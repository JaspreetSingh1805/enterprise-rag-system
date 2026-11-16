from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_metadata_store
from app.api.middleware import TraceIDMiddleware
from app.api.routes import configs, documents, health, jobs, knowledge_bases, query
from app.core.config import settings
from app.core.exceptions import RAGException
from app.core.logging import get_current_trace_id, get_logger, setup_logging

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup structured logging
    setup_logging(log_level=settings.log_level)
    logger.info("Initializing Enterprise RAG Platform...")

    # Ensure default metadata and config v1 are initialized
    store = get_metadata_store()
    active_cfg = store.get_active_config()
    logger.info(f"RAG Platform started with active pipeline config: {active_cfg.version}")

    yield

    logger.info("Shutting down Enterprise RAG Platform...")


app = FastAPI(
    title="Configurable Enterprise RAG Platform",
    description="Production-grade, dynamically configurable RAG system with multi-tenant KBs, versioned pipelines, and grounded cited answers.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(TraceIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
            "trace_id": get_current_trace_id(),
        },
    )


# API Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(knowledge_bases.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(configs.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Configurable Enterprise RAG Platform API is running.",
        "documentation": "/docs",
        "health": "/api/v1/health",
        "version": "1.0.0",
    }
