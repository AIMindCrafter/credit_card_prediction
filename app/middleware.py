"""
middleware.py
~~~~~~~~~~~~~
Custom middleware: structured request logging and request-ID injection.
"""

from __future__ import annotations

import time
import uuid

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request/response with:
    - request_id (injected into response headers)
    - method, path, status_code
    - latency_ms
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        t0 = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                f"[unhandled_exception] request_id={request_id} "
                f"method={request.method} path={request.url.path} error={exc!r}"
            )
            raise

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            f"[request] {request.method} {request.url.path} "
            f"status={response.status_code} latency={latency_ms}ms "
            f"id={request_id[:8]} ip={client_ip}"
        )

        return response
