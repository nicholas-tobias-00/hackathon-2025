"""Task router Flask service for web app integration.

This module distills the notebook prototype from `nicholas_task_classifier.ipynb`
into a reusable Flask API that Next.js (or any frontend) can call. It exposes
REST endpoints for routing-only decisions as well as full downstream model
invocation with fallbacks, structured logging, and optional LangSmith trace
export.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Local helper imports (add ../core to path for notebook parity)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from react_agent.holistic_ai_bedrock import (  # type: ignore  # noqa: E402
    HolisticAIBedrockChat,
    get_chat_model,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TaskRoutingDecision(BaseModel):
    task_type: str = Field(description="High-level classification of the user task")
    recommended_model_family: str = Field(
        description="Which family/type of model should handle the task"
    )
    recommended_model_id: Optional[str] = Field(
        default=None, description="Concrete model identifier to invoke"
    )
    reason: str = Field(description="Step-by-step justification for the decision")
    signals_used: List[str] = Field(
        default_factory=list,
        description="Signals detected in the user's query",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in this routing decision"
    )


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
MODEL_SERIES: Dict[str, Dict[str, Any]] = {
    "anthropic_claude": {
        "label": "Anthropic Claude Series",
        "models": [
            {
                "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "tier": "Recommended",
                "notes": "Balanced depth vs. latency",
                "capabilities": ["big_general", "reasoning"],
            },
            {
                "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                "tier": "Fast",
                "notes": "Lightweight for routing + summaries",
                "capabilities": ["small_fast"],
            },
            {
                "model_id": "us.anthropic.claude-3-opus-20240229-v1:0",
                "tier": "Most Powerful",
                "notes": "High-stakes reasoning",
                "capabilities": ["reasoning", "big_general"],
            },
            {
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "tier": "Balanced",
                "notes": "Previous-gen sonnet",
                "capabilities": ["big_general"],
            },
            {
                "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
                "tier": "Fastest",
                "notes": "Ultra-low latency options",
                "capabilities": ["small_fast"],
            },
            {
                "model_id": "us.anthropic.claude-opus-4-20250514-v1:0",
                "tier": "Cutting Edge",
                "notes": "Latest Claude Opus 4 generation",
                "capabilities": ["reasoning"],
            },
            {
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "tier": "Cutting Edge",
                "notes": "Latest Claude Sonnet 4 generation",
                "capabilities": ["big_general"],
            },
            {
                "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "tier": "Latest",
                "notes": "Claude Sonnet 4.5 preview",
                "capabilities": ["reasoning", "big_general"],
            },
            {
                "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "tier": "Latest Fast",
                "notes": "Claude Haiku 4.5 preview",
                "capabilities": ["small_fast"],
            },
        ],
    },
    "meta_llama": {
        "label": "Meta Llama Series",
        "models": [
            {
                "model_id": "us.meta.llama3-2-90b-instruct-v1:0",
                "tier": "Large",
                "notes": "90B instruction tuned",
                "capabilities": ["big_general", "reasoning"],
            },
            {
                "model_id": "us.meta.llama3-2-11b-instruct-v1:0",
                "tier": "Balanced",
                "notes": "11B for math + analysis",
                "capabilities": ["math", "data_analysis"],
            },
            {
                "model_id": "us.meta.llama3-2-3b-instruct-v1:0",
                "tier": "Lightweight",
                "notes": "3B edge-friendly",
                "capabilities": ["small_fast"],
            },
            {
                "model_id": "us.meta.llama3-2-1b-instruct-v1:0",
                "tier": "Ultra-light",
                "notes": "1B for ultra low-cost",
                "capabilities": ["small_fast"],
            },
            {
                "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
                "tier": "Coding+",
                "notes": "Great for multi-file coding",
                "capabilities": ["coding", "big_general"],
            },
            {
                "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
                "tier": "Coding Fast",
                "notes": "Smaller coding helper",
                "capabilities": ["coding", "small_fast"],
            },
            {
                "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
                "tier": "Next Gen",
                "notes": "Latest Llama 3.3",
                "capabilities": ["big_general", "reasoning"],
            },
            {
                "model_id": "us.meta.llama4-scout-17b-instruct-v1:0",
                "tier": "Scout",
                "notes": "Strong for analytics / scouting",
                "capabilities": ["data_analysis"],
            },
            {
                "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
                "tier": "Maverick",
                "notes": "Advanced math + planning",
                "capabilities": ["math", "reasoning"],
            },
        ],
    },
    "mistral": {
        "label": "Mistral Series",
        "models": [
            {
                "model_id": "us.mistral.pixtral-large-2502-v1:0",
                "tier": "Large",
                "notes": "Pixtral multimodal reasoning",
                "capabilities": ["reasoning", "coding"],
            },
            {
                "model_id": "mistral.mistral-large-2402-v1:0",
                "tier": "General Large",
                "notes": "Great for coding or plans",
                "capabilities": ["coding", "reasoning"],
            },
            {
                "model_id": "mistral.mistral-small-2402-v1:0",
                "tier": "Fast",
                "notes": "Efficient mini-model",
                "capabilities": ["small_fast"],
            },
            {
                "model_id": "mistral.mistral-7b-instruct-v0:2",
                "tier": "Compact",
                "notes": "Open 7B instruct",
                "capabilities": ["small_fast", "coding"],
            },
            {
                "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
                "tier": "Mixture",
                "notes": "Mixture-of-experts for coding",
                "capabilities": ["coding", "reasoning"],
            },
        ],
    },
    "deepseek": {
        "label": "DeepSeek Series",
        "models": [
            {
                "model_id": "us.deepseek.r1-v1:0",
                "tier": "Latest",
                "notes": "DeepSeek R1 reasoning beta",
                "capabilities": ["reasoning"],
            }
        ],
    },
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
for series_key, series in MODEL_SERIES.items():
    for model in series["models"]:
        MODEL_REGISTRY[model["model_id"]] = {
            **model,
            "series_key": series_key,
            "series_label": series["label"],
        }

MODEL_FAMILY_PREFERENCES: Dict[str, List[str]] = {
    "small_fast": [
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us.meta.llama3-2-3b-instruct-v1:0",
        "us.meta.llama3-2-1b-instruct-v1:0",
        "mistral.mistral-small-2402-v1:0",
    ],
    "big_general": [
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.meta.llama3-2-90b-instruct-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
    ],
    "reasoning": [
        "us.anthropic.claude-3-opus-20240229-v1:0",
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "us.deepseek.r1-v1:0",
        "us.mistral.pixtral-large-2502-v1:0",
    ],
    "coding": [
        "us.meta.llama3-1-70b-instruct-v1:0",
        "us.meta.llama3-1-8b-instruct-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
    ],
    "math": [
        "us.meta.llama3-2-11b-instruct-v1:0",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    ],
    "data_analysis": [
        "us.meta.llama4-scout-17b-instruct-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    ],
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
SIGNAL_FAMILY_OVERRIDES: List[Tuple[str, str]] = [
    ("math_problem", "math"),
    ("contains_code_block", "coding"),
    ("mentions_dataframe", "data_analysis"),
    ("long_query", "big_general"),
]


def resolve_model_family(decision: TaskRoutingDecision) -> str:
    for signal, family in SIGNAL_FAMILY_OVERRIDES:
        if signal in decision.signals_used and family in MODEL_FAMILY_PREFERENCES:
            return family
    if decision.recommended_model_family in MODEL_FAMILY_PREFERENCES:
        return decision.recommended_model_family
    if decision.task_type in MODEL_FAMILY_PREFERENCES:
        return decision.task_type
    return "small_fast"


def pick_model_id_for_family(family: str) -> str:
    candidate_ids = MODEL_FAMILY_PREFERENCES.get(family) or []
    if not candidate_ids:
        candidate_ids = MODEL_FAMILY_PREFERENCES.get("small_fast", [])
    for model_id in candidate_ids:
        if model_id in MODEL_REGISTRY:
            return model_id
    raise ValueError("MODEL_REGISTRY is missing a fallback mapping")


def describe_model_choice(model_id: str) -> Dict[str, Any]:
    return MODEL_REGISTRY.get(
        model_id,
        {
            "model_id": model_id,
            "series_label": "Unknown",
            "tier": "unknown",
            "notes": "Model not present in catalog",
        },
    )


def count_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text.split()))


def count_message_tokens(messages: List[Any]) -> Dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0}
    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type in {"HumanMessage", "SystemMessage", "ToolMessage"}:
            totals["input_tokens"] += count_tokens(getattr(msg, "content", ""))
        elif msg_type == "AIMessage":
            totals["output_tokens"] += count_tokens(getattr(msg, "content", ""))
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_serialize(data: Any) -> Any:
    return json.loads(json.dumps(data, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))


# ---------------------------------------------------------------------------
# Router service implementation
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """
You are a SMALL routing model.

Your job:
1. Read the USER'S REQUEST.
2. Decide which downstream model family should handle it:
   - "small_fast": short, simple, casual queries; low risk; no deep reasoning.
   - "big_general": long, multi-step, or high-stakes queries; complex instructions.
   - "reasoning": tasks needing deliberate multi-step reasoning (math, planning, debugging).
   - "coding": code generation, debugging, or explaining code.
   - "math": symbolic math, quantitative finance, stats-heavy work.
   - "data_analysis": CSV / dataframe reasoning, analytics, SQL-style queries.

Rules:
- YOU DO NOT SOLVE THE USER PROBLEM.
- Only classify the task, recommend a model family + specific model, and explain why.

Output:
- Return JSON that matches the TaskRoutingDecision schema exactly.
- In `reason`, walk through the clues you used.
- In `signals_used`, list signals such as 'contains_code_block', 'mentions_bug', 'long_query', etc.
""".strip()


@dataclass
class RouterConfig:
    team_id: str = field(default_factory=lambda: os.environ["HOLISTIC_AI_TEAM_ID"])
    api_token: str = field(default_factory=lambda: os.environ["HOLISTIC_AI_API_TOKEN"])
    router_model_id: str = "us.meta.llama3-2-1b-instruct-v1:0"
    temperature: float = 0.0
    max_tokens: int = 512
    log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ROUTER_LOG_DIR", "submission/decision_logs"))
    )
    fallback_models: List[str] = field(
        default_factory=lambda: [
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "us.meta.llama3-2-3b-instruct-v1:0",
            "us.meta.llama3-2-1b-instruct-v1:0",
            "mistral.mistral-small-2402-v1:0",
        ]
    )


class TaskRouterService:
    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        self.config = config or RouterConfig()
        self.log_dir = self.config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.router_llm = HolisticAIBedrockChat(
            team_id=self.config.team_id,
            api_token=self.config.api_token,
            model=self.config.router_model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", DEFAULT_SYSTEM_PROMPT), ("human", "{user_query}")]
        )
        self.router_chain = prompt | self.router_llm.with_structured_output(
            TaskRoutingDecision
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def route_task(
        self,
        user_query: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TaskRoutingDecision, Optional[Dict[str, Any]]]:
        raw_decision = self.router_chain.invoke({"user_query": user_query})
        router_messages = [
            SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
        token_usage = count_message_tokens(router_messages)
        resolved_family = resolve_model_family(raw_decision)
        resolved_model_id = pick_model_id_for_family(resolved_family)
        enriched_decision = raw_decision.model_copy(
            update={
                "recommended_model_family": resolved_family,
                "recommended_model_id": resolved_model_id,
            }
        )

        trace_payload = None
        if capture_trace:
            trace_payload = {
                "id": os.getenv("LANGSMITH_TRACE_ID") or os.urandom(8).hex(),
                "name": "router.select_model",
                "run_type": "chain",
                "inputs": {"user_query": user_query[:2000]},
                "outputs": {
                    "task_type": enriched_decision.task_type,
                    "model_family": enriched_decision.recommended_model_family,
                    "model_id": enriched_decision.recommended_model_id,
                },
                "extra": {
                    "signals_used": enriched_decision.signals_used,
                    "reason": enriched_decision.reason,
                    "confidence": enriched_decision.confidence,
                },
                "metadata": metadata or {},
                "start_time": utc_now(),
                "end_time": utc_now(),
                "token_usage": token_usage,
            }
        return enriched_decision, trace_payload

    def answer_with_routed_model(
        self,
        user_query: str,
        *,
        system_message: str = "You are a helpful assistant.",
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decision, trace_payload = self.route_task(
            user_query, capture_trace=capture_trace, metadata=metadata
        )
        base_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_query),
        ]
        models_to_try = list(
            dict.fromkeys([decision.recommended_model_id, *self.config.fallback_models])
        )

        response = None
        response_tokens = {}
        last_error: Optional[Exception] = None
        for model_id in models_to_try:
            try:
                downstream_llm = get_chat_model(model_id)
                response = downstream_llm.invoke(base_messages)
                response_tokens = count_message_tokens(base_messages + [response])
                if model_id != decision.recommended_model_id:
                    original = decision.recommended_model_id
                    decision = decision.model_copy(
                        update={
                            "recommended_model_id": model_id,
                            "reason": decision.reason
                            + f" [Fallback: {original} unavailable, switched to {model_id}]",
                            "confidence": decision.confidence * 0.8,
                            "signals_used": decision.signals_used
                            + [f"fallback_from_{original}"],
                        }
                    )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logging.warning("Model %s failed: %s", model_id, exc)
        if response is None:
            raise RuntimeError("All fallback models failed") from last_error

        log_payload = {
            "id": trace_payload.get("id") if trace_payload else os.urandom(8).hex(),
            "timestamp": utc_now(),
            "user_query": user_query,
            "decision": decision.model_dump(),
            "response": getattr(response, "content", None),
            "token_usage": response_tokens,
            "metadata": metadata or {},
        }
        self._append_jsonl("final_outputs.jsonl", log_payload)
        self._write_json("final_output.json", log_payload)

        return {
            "decision": decision.model_dump(),
            "response": getattr(response, "content", None),
            "token_usage": response_tokens,
            "trace": json_serialize(trace_payload) if trace_payload else None,
            "model_metadata": describe_model_choice(decision.recommended_model_id),
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _append_jsonl(self, filename: str, payload: Dict[str, Any]) -> None:
        path = self.log_dir / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_serialize(payload), ensure_ascii=False) + "\n")

    def _write_json(self, filename: str, payload: Dict[str, Any]) -> None:
        path = self.log_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(json_serialize(payload), fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    service = TaskRouterService()
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.get("/health")
    def health_check():
        return {"status": "ok", "time": utc_now().isoformat()}

    @app.get("/api/catalog")
    def catalog():
        return jsonify({
            "families": MODEL_FAMILY_PREFERENCES,
            "models": MODEL_REGISTRY,
            "router_model": service.config.router_model_id,
        })

    @app.post("/api/route")
    def route_endpoint():
        body = request.get_json(silent=True) or {}
        user_query = body.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "query is required"}), 400
        capture_trace = bool(body.get("captureTrace", True))
        metadata = body.get("metadata") or {}
        decision, trace_payload = service.route_task(
            user_query, capture_trace=capture_trace, metadata=metadata
        )
        return jsonify(
            {
                "decision": decision.model_dump(),
                "trace": json_serialize(trace_payload) if trace_payload else None,
                "model_metadata": describe_model_choice(
                    decision.recommended_model_id or "unknown"
                ),
            }
        )

    @app.post("/api/answer")
    def answer_endpoint():
        body = request.get_json(silent=True) or {}
        user_query = body.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "query is required"}), 400
        system_message = body.get("systemMessage", "You are a helpful assistant.")
        capture_trace = bool(body.get("captureTrace", True))
        metadata = body.get("metadata") or {}
        try:
            result = service.answer_with_routed_model(
                user_query,
                system_message=system_message,
                capture_trace=capture_trace,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover
            logging.exception("answer endpoint failed")
            return jsonify({"error": str(exc)}), 500
        return jsonify(result)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("ROUTER_LOG_LEVEL", "INFO"))
    port = int(os.getenv("PORT", "8000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port)
