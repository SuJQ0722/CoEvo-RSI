"""Who&When-specific MetaAgent direction."""

from collections.abc import Mapping
from typing import Any


def build(config: Mapping[str, Any]) -> str:
    selection = config.get("selection") or {}
    metrics = ", ".join(selection.get(
        "quality_metrics", ["agent_accuracy", "step_accuracy"]
    ))
    groups = selection.get("ranking_metric_groups")
    weights = selection.get("ranking_metric_weights") or {}
    objective = (
        "Ranking objectives, in priority order: "
        + " > ".join("weighted_mean(" + ", ".join(
            f"{name}: weight={weights.get(name, 1)}" for name in group
        ) + ")" for group in groups)
        + ". Optimize the configured joint objective."
        if groups else f"Ranking follows the configured quality metric order: {metrics}."
    )
    return f"""Domain: Who&When failure attribution.
Validation admission metrics: {metrics}.
{objective}
Diagnose responsible-agent and decisive-step errors from the validation evidence.
Analyze validation sources separately, then optimize the joint weighted
objective above. Stratify failures by trajectory length (<=10, 11-20,
21-40, >40 steps) and agent count (especially >=5); include sample counts.
Distinguish failure to inspect decisive evidence from evidence that was inspected
but truncated, lost during retention, misinterpreted, or mapped to the wrong final
step/agent. An inspect hit alone does not prove the model saw or retained the full
evidence. Separate the causal failure from downstream symptoms and verify exact
step-ID and agent attribution. Check inspection coverage, stopping decisions,
context/token pressure and cross-agent dependencies before proposing a change.
When training evidence is available, compare the SAME harness/model initialization
before and after SFT on each validation source. Check unique long/multi-agent training cases,
not just repeated row counts, and inspect training reasoning, retained decisive
evidence, supervision and training truncation. Joint post-SFT regression is not
merely a tradeoff between validation sources. Do not assume data curation or more repetition fixes it.
Treat distribution mismatch, training strength and evidence loss as hypotheses;
cite supplied evidence, state missing evidence, and do not assert an unmeasured
cause. Keep any proposed training changes within the stage's authorized scope.
Carry the leading supported failure mechanism, a targeted change, and measurable
validation checks into the shared context and candidate briefs. Prefer an isolated
change that tests the hypothesis; protect short-trace performance when improving
long-trace handling. Never hardcode validation answers or sample-specific rules.
Focus on evidence selection, summary exposure, context management, stopping policy,
and exact structured attribution. Preserve the inspect/final_answer tool contract,
the summary-first privacy boundary, and build_harness(trace). Never expose labels
or full private step text before inspect is called."""
