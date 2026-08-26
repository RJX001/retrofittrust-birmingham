"""FastAPI integration backend — rank, explain, ledger, twin write-back.

Not a production service: no auth (explicit non-goal). Orchestrates the
twin → AI → ledger → SQLite loop from CURSOR_BUILD_SPEC §6.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from retrofittrust.api.features import (
    SHAP_CAVEAT,
    load_lsoa_frame,
    load_model_bundle,
    predict_priority,
    shap_for_row,
)
from retrofittrust.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    LedgerAppendRequest,
    LedgerAppendResponse,
    LedgerVerifyResponse,
    RankItem,
    RankRequest,
    RankResponse,
    ShapFeature,
)
from retrofittrust.config import EPC_GAP_WEIGHT, IMD_INCOME_WEIGHT, LEDGER_PATH, SQLITE_PATH
from retrofittrust.ledger.chain import Ledger, load_or_create
from retrofittrust.ledger.synthetic import SYNTHETIC_LABEL, generate_event
from retrofittrust.ledger.tamper import demonstrate_tampering
from retrofittrust.ledger.twin_state import (
    apply_ledger_event,
    cache_priority_scores,
    fetch_all_lsoa_state,
    init_twin_db,
    save_cohort_selection,
)


def _frame_by_codes(codes: list[str]):
    frame, _source = load_lsoa_frame(allow_synthetic_fallback=True)
    subset = frame[frame["lsoa21cd"].isin([str(c) for c in codes])].copy()
    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=f"None of the requested LSOA codes were found. Example codes: {frame['lsoa21cd'].head(5).tolist()}",
        )
    return subset, frame


def _quality_flag(row) -> str | None:
    flag = row.get("anomaly_flag")
    if flag is None:
        return None
    try:
        if float(flag) > 0:
            return "low_confidence"
    except (TypeError, ValueError):
        if str(flag).lower() in {"true", "1", "flagged"}:
            return "low_confidence"
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_twin_db(SQLITE_PATH)
    load_or_create(LEDGER_PATH)
    yield


app = FastAPI(
    title="RetrofitTrust Birmingham API",
    description=(
        "PoC orchestration for ranking, SHAP explanations, and a SHA-256 hash-chain "
        "ledger. Grant/verification payloads are SYNTHETIC DATA."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PoC only — production auth is an explicit non-goal
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = load_model_bundle()
    ok, msg = Ledger(LEDGER_PATH).verify_chain()
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "ledger_valid": ok,
        "ledger_message": msg,
        "sqlite_path": str(SQLITE_PATH),
        "ledger_path": str(LEDGER_PATH),
    }


def _rank_via_program1(codes: list[str]) -> list[RankItem] | None:
    """Prefer the trained LightGBM artefact when Program 1 has produced it."""
    try:
        from retrofittrust.modeling.predict import rank_lsoas
    except ImportError:
        return None
    try:
        result = rank_lsoas(lsoa_codes=codes)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    twin = fetch_all_lsoa_state()
    items: list[RankItem] = []
    for rank_i, row in enumerate(result.get("rankings") or [], start=1):
        code = str(row.get("lsoa21cd") or row.get("lsoa_code") or row.get("lsoa"))
        state = twin.get(code, {})
        live = state.get("priority_score")
        score = float(live) if live is not None else float(row.get("score") or 0.0)
        low = bool(row.get("low_confidence"))
        items.append(
            RankItem(
                lsoa21cd=code,
                lsoa21nm=None,
                priority_score=score,
                rank=rank_i,
                data_quality_flag="low_confidence" if low else None,
                verified=bool(state.get("verified")),
            )
        )
    return items or None


@app.post("/rank", response_model=RankResponse)
def rank(payload: RankRequest) -> RankResponse:
    p1_items = _rank_via_program1(payload.lsoa_codes)
    if p1_items is not None:
        if payload.top_k:
            p1_items = p1_items[: payload.top_k]
        cache_priority_scores(
            [{"lsoa_code": it.lsoa21cd, "priority_score": it.priority_score} for it in p1_items],
            source="model",
        )
        save_cohort_selection(
            [it.lsoa21cd for it in p1_items],
            metadata={"endpoint": "/rank", "n": len(p1_items), "source": "lightgbm"},
        )
        return RankResponse(
            items=p1_items,
            model_loaded=True,
            source="lightgbm",
            notes=[
                "Priority scores are for relative ranking, not absolute energy prediction.",
                "Ecological fallacy: LSOA IMD does not describe individual households.",
                "Flagged records are scored with a low-confidence caveat — never silently deleted.",
            ],
        )

    subset, _full = _frame_by_codes(payload.lsoa_codes)
    bundle = load_model_bundle()
    scores = predict_priority(subset, bundle)
    subset = subset.copy()
    subset["priority_score"] = scores
    subset = subset.sort_values("priority_score", ascending=False)
    if payload.top_k:
        subset = subset.head(payload.top_k)

    twin = fetch_all_lsoa_state()
    items: list[RankItem] = []
    cache_rows: list[dict[str, Any]] = []
    for rank_i, (_, row) in enumerate(subset.iterrows(), start=1):
        code = str(row["lsoa21cd"])
        state = twin.get(code, {})
        live_score = state.get("priority_score")
        score = float(live_score) if live_score is not None else float(row["priority_score"])
        items.append(
            RankItem(
                lsoa21cd=code,
                lsoa21nm=None if pd.isna(row.get("lsoa21nm")) else str(row.get("lsoa21nm")),
                priority_score=score,
                rank=rank_i,
                data_quality_flag=_quality_flag(row),
                verified=bool(state.get("verified")),
            )
        )
        cache_rows.append({"lsoa_code": code, "priority_score": score})

    save_cohort_selection(
        [it.lsoa21cd for it in items],
        metadata={"endpoint": "/rank", "n": len(items)},
    )
    cache_priority_scores(
        cache_rows,
        source="model" if bundle is not None else "composite",
    )

    notes = [
        "Priority scores are for relative ranking, not absolute energy prediction.",
        "Ecological fallacy: LSOA IMD does not describe individual households.",
    ]
    if bundle is None:
        notes.append(
            "models/ranking_model.joblib not found — using composite weights "
            f"(EPC gap {EPC_GAP_WEIGHT:.0%} / IMD income {IMD_INCOME_WEIGHT:.0%})."
        )
    return RankResponse(
        items=items,
        model_loaded=bundle is not None,
        source="lightgbm" if bundle is not None else "composite",
        notes=notes,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest) -> ExplainResponse:
    try:
        from retrofittrust.modeling.explain import explain_lsoa

        p1 = explain_lsoa(lsoa_code=payload.lsoa21cd, top_k=payload.top_n)
        contrib = p1.get("top_contributions") or []
        return ExplainResponse(
            lsoa21cd=str(payload.lsoa21cd),
            base_value=float(p1.get("base_value") or 0.0),
            prediction=float(p1.get("prediction") or 0.0),
            features=[
                ShapFeature(
                    feature=str(c.get("feature")),
                    value=c.get("value"),
                    shap_value=float(c.get("shap_value") or 0.0),
                )
                for c in contrib
            ],
            method=str(p1.get("method") or "shap_tree_explainer"),
            caveat=str(p1.get("caveat") or SHAP_CAVEAT),
            model_loaded=True,
        )
    except (FileNotFoundError, ValueError, ImportError, KeyError):
        pass

    subset, _full = _frame_by_codes([payload.lsoa21cd])
    row = subset.iloc[[0]]
    bundle = load_model_bundle()
    result = shap_for_row(row, bundle, top_n=payload.top_n)
    return ExplainResponse(
        lsoa21cd=str(payload.lsoa21cd),
        base_value=float(result["base_value"]),
        prediction=float(result["prediction"]),
        features=[ShapFeature(**f) for f in result["features"]],
        method=str(result["method"]),
        caveat=str(result.get("caveat") or SHAP_CAVEAT),
        model_loaded=bool(result.get("model_loaded")),
    )


@app.post("/ledger/append", response_model=LedgerAppendResponse)
def ledger_append(payload: LedgerAppendRequest) -> LedgerAppendResponse:
    details = dict(payload.details)
    synthetic = payload.generate_synthetic
    if synthetic:
        generated = generate_event(
            payload.event_type,
            payload.lsoa21cd,
            priority_score=payload.priority_score or float(details.get("priority_score") or 0.0),
            epc_uplift_bands=payload.epc_uplift_bands,
            extra=details,
        )
        data = generated
    else:
        data = {
            "type": payload.event_type,
            "lsoa": payload.lsoa21cd,
            "label": SYNTHETIC_LABEL,
            **details,
        }

    block = load_or_create(LEDGER_PATH).append_block(data, persist=True)
    apply_ledger_event(
        str(data.get("type") or payload.event_type),
        payload.lsoa21cd,
        data,
        ledger_index=int(block["index"]),
    )
    ok, _msg = Ledger(LEDGER_PATH).verify_chain()
    return LedgerAppendResponse(
        block=block,
        chain_valid=ok,
        twin_state_updated=True,
        synthetic=synthetic,
    )


@app.get("/ledger/verify", response_model=LedgerVerifyResponse)
def ledger_verify() -> LedgerVerifyResponse:
    ledger = Ledger(LEDGER_PATH)
    ok, errors = ledger.verify_chain_detailed()
    message = f"ok ({len(ledger.chain)} blocks)" if ok else "; ".join(errors)
    return LedgerVerifyResponse(
        valid=ok,
        length=len(ledger.chain),
        message=message,
        errors=errors,
        recent_blocks=ledger.recent_blocks(8),
    )


@app.get("/ledger/tamper-demo")
def ledger_tamper_demo() -> dict[str, Any]:
    """In-memory tamper demo — does not persist. Dissertation evidence only."""
    try:
        return demonstrate_tampering(Ledger(LEDGER_PATH))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
