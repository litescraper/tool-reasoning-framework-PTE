"""
FastAPI service for Serper (Google Search) and Jina (Webpage Content) tools.

This service provides endpoints for:
- /serper: Google Search via Serper API
- /jina: Webpage content extraction via Jina API

Usage:
    Set environment variables:
    - SERPER_API_KEY: Your Serper API key
    - JINA_API_KEY: Your Jina API key
    
    Then run:
    python services/fastapi_serper_jina.py
    or
    uvicorn services.fastapi_serper_jina:app --host 0.0.0.0 --port 5003 --workers 4
"""
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Any, Dict, Optional
import random
import httpx
import time
import requests
import logging
import json
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Keys (set via environment variables)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

if not SERPER_API_KEY:
    logger.warning("SERPER_API_KEY not set. Search functionality may not work.")
if not JINA_API_KEY:
    logger.warning("JINA_API_KEY not set. Jina functionality may not work.")

app = FastAPI(title="Tool Reasoning Framework - Search & Visit Service")

# Service startup timestamp
_service_startup_timestamp: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Record service startup timestamp"""
    global _service_startup_timestamp
    _service_startup_timestamp = time.time()
    logger.info(f"🚀 Service started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_service_startup_timestamp))}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "uptime": time.time() - _service_startup_timestamp if _service_startup_timestamp else 0}

@app.post("/serper")
async def serper_api(request: Request):
    """
    Serper API proxy endpoint.
    
    Supports both JSON and form-encoded requests:
    - JSON: {"query": "..."} or {"q": "..."}
    - Form: query=... or q=...
    
    Returns Serper API response with search results.
    """
    payload: Dict[str, Any] = {}
    try:
        payload = await request.json()
    except Exception:
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    q = payload.get("query") or payload.get("q")
    if not q:
        raise HTTPException(status_code=400, detail="missing 'query' (or 'q') in request body")

    # Map 'query' to 'q' for Serper API
    payload.pop("query", None)
    payload.setdefault("q", q)

    st = time.time()
    logger.info(f"📢 [/serper] query={q}")

    max_retries = 3
    last_exception = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f'{os.getenv("SERPER_BASE_URL", "https://google.serper.dev").rstrip("/")}/search',
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            r.raise_for_status()
            elapsed = time.time() - st
            logger.info(f"✅ [/serper] status={r.status_code} elapsed={elapsed:.3f}s")
            
            return JSONResponse(content=r.json(), status_code=200)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ [/serper] attempt {attempt + 1}/{max_retries} failed, retrying... Error: {str(e)}")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff: 0.5s, 1s, 1.5s
            else:
                logger.exception(f"❌ [/serper] error after {max_retries} attempts")
    
    return JSONResponse(content={"error": str(last_exception)}, status_code=502)

@app.api_route("/jina", methods=["GET", "POST"])
async def jina_api(request: Request):
    """
    Jina API proxy endpoint for webpage content extraction.
    
    Supports multiple request formats:
    - Query string: /jina?url=...
    - JSON: {"url": "..."}
    - Form: url=...
    
    Returns webpage content as plain text.
    """
    url = request.query_params.get("url")
    if not url:
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                url = payload.get("url")
        except Exception:
            try:
                form = await request.form()
                url = form.get("url")
            except Exception:
                url = None

    t0 = time.time()
    logger.info(f"🔗 [/jina] url={url}")
    
    if not url:
        return PlainTextResponse("missing url", status_code=400)
    
    max_retries = 3
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[JINA] Starting request for URL: {url} (attempt {attempt + 1}/{max_retries})")
            time.sleep(random.uniform(0.5, 1))  # Random delay to avoid rate limiting
            r = requests.get(
                f"https://r.jina.ai/{url}",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                timeout=20
            )
            r.raise_for_status()
            elapsed = time.time() - t0
            logger.info(f"[JINA] Completed in {elapsed:.2f}s")
            
            return PlainTextResponse(r.text, status_code=r.status_code)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ [JINA] attempt {attempt + 1}/{max_retries} failed, retrying... Error: {str(e)}")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f'[JINA] Failed after {time.time() - t0:.2f}s after {max_retries} attempts')
    
    return PlainTextResponse(str(last_exception), status_code=502)


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", "5003"))
    workers = int(os.getenv("SERVICE_WORKERS", "4"))
    logger.info(f"Starting service on port {port} with {workers} workers")
    uvicorn.run("services.fastapi_serper_jina:app", host="0.0.0.0", port=port, workers=workers)

