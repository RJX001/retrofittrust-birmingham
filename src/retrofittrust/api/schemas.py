"""Pydantic request/response schemas for the FastAPI integration backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["eligibility", "works_claimed", "verification"]


class RankRequest(BaseModel):
    lsoa_codes: list[str] = Field(..., min_length=1, description="2021 LSOA codes to rank")
    top_k: int | None = Field(default=None, ge=1, description="Optional truncation after ranking")


class RankItem(BaseModel):
    lsoa21cd: str
    lsoa21nm: str | None = None
    priority_score: float
    rank: int
    data_quality_flag: str | None = None
    verified: bool = False


class RankResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: list[RankItem]
    model_loaded: bool
    source: str
    notes: list[str] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    lsoa21cd: str
    top_n: int = Field(default=10, ge=1, le=30)


class ShapFeature(BaseModel):
    feature: str
    value: float | str | None = None
    shap_value: float


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    lsoa21cd: str
    base_value: float
    prediction: float
    features: list[ShapFeature]
    method: str
    caveat: str
    model_loaded: bool


class LedgerAppendRequest(BaseModel):
    event_type: EventType
    lsoa21cd: str
    details: dict[str, Any] = Field(default_factory=dict)
    generate_synthetic: bool = Field(
        default=True,
        description="Fill grant/works/verification fields from the SYNTHETIC DATA generator",
    )
    priority_score: float | None = None
    epc_uplift_bands: int = Field(default=2, ge=0, le=6)


class LedgerAppendResponse(BaseModel):
    block: dict[str, Any]
    chain_valid: bool
    twin_state_updated: bool
    synthetic: bool


class LedgerVerifyResponse(BaseModel):
    valid: bool
    length: int
    message: str
    errors: list[str] = Field(default_factory=list)
    recent_blocks: list[dict[str, Any]] = Field(default_factory=list)
