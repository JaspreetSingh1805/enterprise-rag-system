import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import set_current_trace_id


class TraceIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts or generates an X-Trace-ID for every incoming HTTP request,
    stores it in contextvars for structured logging, and attaches it to response headers.
    Also measures request latency in milliseconds.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
        set_current_trace_id(trace_id)

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        return response
