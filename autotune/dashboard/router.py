"""Dashboard FastAPI router — /dashboard UI and /api/dashboard/* JSON endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["dashboard"])

_STATIC_DIR = Path(__file__).parent / "static"


@router.get("/dashboard", include_in_schema=False)
async def dashboard_ui():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return HTMLResponse("<h1>Dashboard static files not found</h1>", status_code=500)


@router.get("/api/dashboard/overview")
async def dashboard_overview():
    from .metrics import get_overview
    return get_overview()


@router.get("/api/dashboard/requests")
async def dashboard_requests():
    from .metrics import get_requests_timeseries
    return {"data": get_requests_timeseries()}


@router.get("/api/dashboard/ttft_trend")
async def dashboard_ttft_trend():
    from .metrics import get_ttft_trend
    return {"data": get_ttft_trend()}


@router.get("/api/dashboard/models")
async def dashboard_models():
    from .metrics import get_models_stats
    return {"models": get_models_stats()}


@router.get("/api/dashboard/comparison")
async def dashboard_comparison():
    from .metrics import get_comparison
    return get_comparison()


@router.get("/api/dashboard/keys")
async def dashboard_keys():
    from .metrics import get_api_keys_summary
    return {"keys": get_api_keys_summary()}


@router.get("/api/dashboard/slow")
async def dashboard_slow():
    from .metrics import get_slow_requests
    return {"requests": get_slow_requests()}


@router.get("/api/dashboard/suggestions")
async def dashboard_suggestions():
    from .metrics import get_suggestions
    return {"suggestions": get_suggestions()}
