# metrics.py
from typing import Dict, Any

# Approximate cost per 1K tokens (USD)
COST_PER_1K = {
    "small_llm": 0.0005,
    "large_llm": 0.0020,
    "classifier": 0.0004,
}

# Approximate energy per 1K tokens (kWh)
KWH_PER_1K = {
    "small_llm": 0.0002,    # smaller model, less compute
    "large_llm": 0.0006,    # bigger model, more compute
    "classifier": 0.00015,
}

# Grid emission factor (gCO2 per kWh), rough global average
CO2_PER_KWH = 400.0  # grams CO2 / kWh


def model_category(model_name: str) -> str:
    """
    Map Holistic model names to categories for cost/emissions.
    Adjust this mapping based on which models you actually use.
    """
    name = model_name.lower()

    # Small models: haiku, small llama, etc.
    if "haiku" in name or "llama3-2-3b" in name or "llama3-2-11b" in name:
        return "small_llm"

    # Large models: sonnet, opus, large llama, nova-pro
    if "sonnet" in name or "opus" in name or "llama3-2-90b" in name or "nova-pro" in name:
        return "large_llm"

    # Classifier / router / anything else
    return "classifier"


def compute_metrics_from_trace(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    total_tokens = 0
    total_cost_actual = 0.0
    total_energy_actual = 0.0  # kWh
    total_emissions_actual = 0.0  # gCO2
    total_latency = 0
    n_steps = len(trace_dict["steps"])

    for step in trace_dict["steps"]:
        tokens = step.get("tokens", 0)
        total_tokens += tokens
        total_latency += step.get("latency_ms", 0)

        cat = model_category(step.get("model", ""))
        cost = (tokens / 1000.0) * COST_PER_1K[cat]
        energy = (tokens / 1000.0) * KWH_PER_1K[cat]
        emissions = energy * CO2_PER_KWH

        total_cost_actual += cost
        total_energy_actual += energy
        total_emissions_actual += emissions

    # Baseline: pretend ALL tokens go to large LLM
    baseline_cat = "large_llm"
    total_cost_baseline = (total_tokens / 1000.0) * COST_PER_1K[baseline_cat]
    total_energy_baseline = (total_tokens / 1000.0) * KWH_PER_1K[baseline_cat]
    total_emissions_baseline = total_energy_baseline * CO2_PER_KWH

    def pct_saving(actual: float, baseline: float) -> float:
        if baseline == 0:
            return 0.0
        return 100.0 * (1.0 - actual / baseline)

    metrics = {
        "totals": {
            "tokens": total_tokens,
            "cost_actual": total_cost_actual,
            "cost_baseline": total_cost_baseline,
            "energy_actual_kwh": total_energy_actual,
            "energy_baseline_kwh": total_energy_baseline,
            "emissions_actual_g": total_emissions_actual,
            "emissions_baseline_g": total_emissions_baseline,
            "avg_latency_ms": total_latency / max(n_steps, 1),
        },
        "savings": {
            "cost_savings_pct": pct_saving(total_cost_actual, total_cost_baseline),
            "energy_savings_pct": pct_saving(total_energy_actual, total_energy_baseline),
            "emissions_savings_pct": pct_saving(
                total_emissions_actual, total_emissions_baseline
            ),
        },
    }
    return metrics


if __name__ == "__main__":
    # quick sanity check
    fake_trace = {
        "query_id": "demo",
        "user_query": "Test",
        "steps": [
            {"model": "claude-3-5-haiku", "tokens": 200, "latency_ms": 300},
            {"model": "claude-3-5-sonnet", "tokens": 400, "latency_ms": 700},
        ],
    }
    print(compute_metrics_from_trace(fake_trace))
