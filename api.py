"""FastAPI HTTP interface for on-demand premarket pipeline execution."""

from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import run


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load environment variables before the app starts accepting requests."""
    yield


app = FastAPI(lifespan=lifespan)


class RunRequest(BaseModel):
    """Request body for POST /run."""

    tickers: list[str] | None = None
    regime: Literal["BULLISH", "BEARISH"] | None = None
    test_mode: bool = True


@app.post("/run")
async def run_pipeline(req: RunRequest):
    """Trigger the premarket pipeline and return strategy cards + gate log as JSON.

    test_mode=True (default) routes LLM calls to Gemini; False uses Claude.
    """
    argv: list[str] = []
    if req.tickers:
        argv.extend(req.tickers)
    if req.regime:
        argv += ["--regime", req.regime]

    try:
        strategy_cards, log_entries = await run.main(argv or None, test_mode=req.test_mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"strategy_cards": strategy_cards, "log_entries": log_entries}
