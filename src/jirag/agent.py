"""Small, read-only orchestration layer for the jiRAG project."""

from __future__ import annotations

import re
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_json_atomic
from .generation import validate_citations


SearchFunction = Callable[[str], list[dict[str, Any]]]
GenerateFunction = Callable[[str, list[dict[str, Any]]], dict[str, Any]]

_ID_PATTERN = re.compile(
    r"(?:tckt\s*[-#:]?\s*|ticket(?:\s+number)?\s*[-#:]?\s*|טיקט\s*[-#:]?\s*)(\d{1,4})",
    re.IGNORECASE,
)
_COUNT_WORDS = ("how many", "count", "number of", "כמה", "מספר")
_OPEN_WORDS = ("open", "remain", "remaining", "פתוח", "פתוחים", "נשאר", "נותר")
_ADVICE_WORDS = (
    "recommend", "should we", "what should", "advise", "suggest",
    "ממליץ", "המלצה", "כדאי", "מה לעשות", "האם צריך",
)
_RISK_WORDS = (
    "security", "sensitive", "exposure", "exposed", "permission", "deleted",
    "deletion", "supplier information", "outside procurement",
    "אבטחה", "רגיש", "חשיפה", "הרשאה", "נמחק", "מידע ספקים", "מחוץ לרכש",
)
_MULTI_WORDS = ("compare", "both", "multiple", "across", "השווה", "שניהם", "כמה תקלות")
_ABSTENTION_PHRASES = (
    "evidence does not resolve", "evidence is insufficient", "insufficient evidence",
    "not enough evidence", "cannot determine", "can't determine", "cannot confirm",
    "no supporting evidence", "אין מספיק ראיות", "הראיות אינן מספיקות",
    "לא נמצאה ראיה", "לא ניתן לקבוע", "לא ניתן לאשר", "המידע אינו מספיק",
)
_PRIORITIES = ("Highest", "High", "Medium", "Low", "Lowest")
_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "there", "these", "this", "to", "was", "were", "what", "when",
    "which", "who", "why", "with",
}


@dataclass(frozen=True)
class AgentPlan:
    """Transparent plan produced before any tool is executed."""

    route: str
    ticket_id: str | None
    semantic_query: str
    filters: dict[str, Any]
    needs_generation: bool
    advisory_level: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "ticket_id": self.ticket_id,
            "semantic_query": self.semantic_query,
            "filters": self.filters,
            "needs_generation": self.needs_generation,
            "advisory_level": self.advisory_level,
            "reason": self.reason,
        }


class AdvisoryAgent:
    """Route questions, call existing Reader tools and return safe advice only."""

    def __init__(
        self,
        *,
        retrieval_engine: Any,
        semantic_search: SearchFunction,
        generate_answer: GenerateFunction,
        minimum_semantic_score: float = 0.70,
    ) -> None:
        self.retrieval = retrieval_engine
        self.semantic_search = semantic_search
        self.generate_answer = generate_answer
        self.minimum_semantic_score = float(minimum_semantic_score)

    @staticmethod
    def _clean(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Agent input must be a non-empty string")
        return text.strip()

    @staticmethod
    def _contains(text: str, phrases: tuple[str, ...]) -> bool:
        lowered = text.casefold()
        return any(phrase.casefold() in lowered for phrase in phrases)

    def _filters(self, question: str) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        lowered = question.casefold()
        matched_priorities = [p for p in _PRIORITIES if re.search(rf"\b{p.casefold()}\b", lowered)]
        if matched_priorities:
            filters["priority"] = matched_priorities[0] if len(matched_priorities) == 1 else matched_priorities
        if self._contains(question, _OPEN_WORDS):
            filters["is_open"] = True
        return filters

    def plan(self, text: str, *, new_ticket: bool = False) -> AgentPlan:
        """Create a compact plan; unseen questions fall back to semantic RAG."""
        question = self._clean(text)
        filters = self._filters(question)
        id_match = _ID_PATTERN.search(question)
        advice_requested = self._contains(question, _ADVICE_WORDS)
        risk_present = self._contains(question, _RISK_WORDS)
        advisory_level = "attention" if risk_present else ("review" if advice_requested else "none")

        if new_ticket:
            return AgentPlan(
                "similarity", None, question, {}, True,
                advisory_level if advisory_level != "none" else "review",
                "A new ticket is compared with documented incidents; advice remains non-executable.",
            )
        if id_match:
            ticket_id = f"tckt-{int(id_match.group(1)):04d}"
            needs_generation = any(
                term in question.casefold()
                for term in ("why", "how", "resolved", "recommend", "למה", "איך", "נפתר", "כדאי")
            )
            return AgentPlan(
                "lookup", ticket_id, question, {}, needs_generation,
                advisory_level, "An explicit ticket ID requires exact lookup.",
            )
        if self._contains(question, _COUNT_WORDS):
            return AgentPlan(
                "aggregation", None, question, filters, False,
                advisory_level, "A numeric question requires deterministic filtering and counting.",
            )
        if filters:
            return AgentPlan(
                "hybrid", None, question, filters, True,
                advisory_level, "The question combines a semantic topic with exact metadata constraints.",
            )
        return AgentPlan(
            "semantic", None, question, {}, True,
            advisory_level, "Free-form incident reasoning uses semantic RAG.",
        )

    def _retrieve(self, plan: AgentPlan) -> tuple[list[dict[str, Any]], list[str], float | None]:
        if plan.route == "lookup":
            result = self.retrieval.lookup_ticket(plan.ticket_id)
            return [result], ["get_ticket"], None
        if plan.route == "aggregation":
            return [], ["filter_tickets", "aggregate_tickets"], None
        if plan.route == "hybrid":
            results = self.retrieval.hybrid_search(plan.semantic_query, plan.filters, top_k=5)
            score = results[0]["score"] if results else None
            return results, ["filter_tickets", "hybrid_search"], score

        raw = self.retrieval.semantic_search(plan.semantic_query, top_k=1)
        score = raw[0]["score"] if raw else None
        results = self.semantic_search(plan.semantic_query)
        tools = ["search_tickets", "rerank_tickets"]
        if plan.route == "similarity":
            tools.append("compare_new_ticket")
        return results, tools, score

    def _evidence_state(
        self, plan: AgentPlan, results: list[dict[str, Any]], score: float | None
    ) -> tuple[str, str, float | None]:
        if plan.route in {"lookup", "aggregation"}:
            return "sufficient", "The result is produced by an exact deterministic tool.", None
        if not results or score is None or score < self.minimum_semantic_score:
            return "insufficient", "No sufficiently similar evidence was retrieved.", None
        if plan.route == "similarity":
            return "partial", "Similar incidents support advice, but do not prove the same root cause.", None
        lexical_support = self._lexical_support(plan.semantic_query, results)
        if self._contains(plan.semantic_query, _MULTI_WORDS) and len(results) < 2:
            return "partial", "The comparative question did not retrieve multiple sources.", lexical_support
        return (
            "sufficient",
            "Relevant Jira evidence passed the semantic pre-check; direct support is diagnostic only.",
            lexical_support,
        )

    @staticmethod
    def _lexical_support(question: str, results: list[dict[str, Any]]) -> float | None:
        """Return a transparent English token-overlap diagnostic, never a hard gate."""
        if any("\u0590" <= character <= "\u05ff" for character in question):
            return None
        query_tokens = {
            token for token in re.findall(r"[a-zA-Z][a-zA-Z-]+", question.casefold())
            if len(token) >= 4 and token not in _ENGLISH_STOPWORDS
        }
        if not query_tokens:
            return None
        evidence_text = " ".join(
            " ".join((
                row.get("content", {}).get("summary", ""),
                row.get("content", {}).get("component", ""),
                row.get("content", {}).get("description", ""),
            ))
            for row in results[:3]
        ).casefold()
        return sum(token in evidence_text for token in query_tokens) / len(query_tokens)

    @staticmethod
    def _is_abstention(answer: str) -> bool:
        """Detect an explicit generator refusal after it has inspected retrieved evidence."""
        return AdvisoryAgent._contains(answer, _ABSTENTION_PHRASES)

    @staticmethod
    def _deterministic_lookup(result: dict[str, Any]) -> str:
        return (
            f"{result['document_id']} is {result['metadata']['status']}: "
            f"{result['content']['summary']} [{result['document_id']}]"
        )

    @staticmethod
    def _advice(
        plan: AgentPlan,
        results: list[dict[str, Any]],
        evidence_state: str,
        *,
        new_ticket: bool,
    ) -> dict[str, Any]:
        advice_requested = AdvisoryAgent._contains(plan.semantic_query, _ADVICE_WORDS)
        risk_present = AdvisoryAgent._contains(plan.semantic_query, _RISK_WORDS)
        trigger = (
            "new_ticket" if new_ticket else
            "user_request" if advice_requested else
            "evidence_risk" if risk_present else
            "none"
        )
        advisory = {
            "level": plan.advisory_level,
            "trigger": trigger,
            "recommendation": None,
            "evidence_ids": [row["document_id"] for row in results],
            "human_approval_required": False,
            "write_executed": False,
        }
        if plan.advisory_level == "none":
            return advisory
        if evidence_state == "insufficient":
            advisory.update({
                "recommendation": "Collect more incident details before choosing a specific action.",
                "human_approval_required": True,
            })
            return advisory
        if plan.route == "similarity":
            advisory.update({
                "recommendation": (
                    "Compare permissions, scope and root cause with the cited incidents before assigning "
                    "priority, escalation or duplicate status."
                ),
                "human_approval_required": True,
            })
            return advisory

        verified_done = bool(results) and all(
            row.get("metadata", {}).get("status") == "Done"
            and row.get("evaluation", {}).get("solution_type") == "solution-verified"
            for row in results
        )
        recommendation = (
            "No reopening is indicated by the verified resolution; confirm that the preventive control remains active."
            if verified_done else
            "Review the cited evidence before deciding whether follow-up or escalation is required."
        )
        advisory.update({
            "recommendation": recommendation,
            "human_approval_required": True,
        })
        return advisory

    @staticmethod
    def _solution_state_safe(
        answer: str,
        results: list[dict[str, Any]],
        cited_ids: list[str],
    ) -> tuple[bool, list[str]]:
        """Reject strong resolution language only when it concerns a cited weak source."""
        cited = {row["document_id"]: row for row in results if row["document_id"] in cited_ids}
        lowered = answer.casefold()
        warnings = []
        for ticket_id, row in cited.items():
            state = row.get("evaluation", {}).get("solution_type", "")
            if state == "solution-unresolved" and any(
                phrase in lowered for phrase in ("was resolved", "has been resolved", "permanent fix", "fully fixed")
            ):
                warnings.append(f"{ticket_id} is unresolved but the answer uses definitive resolution language")
            if state == "solution-workaround" and any(
                phrase in lowered for phrase in ("permanent fix", "fully resolved", "root cause was fixed")
            ):
                warnings.append(f"{ticket_id} documents a workaround but the answer presents a permanent fix")
        return not warnings, warnings

    def run(self, text: str, *, new_ticket: bool = False) -> dict[str, Any]:
        """Execute one read-only agent turn and expose a compact decision trace."""
        plan = self.plan(text, new_ticket=new_ticket)

        if plan.route == "aggregation":
            aggregation = self.retrieval.aggregate_tickets(plan.filters, group_by="status")
            answer = f"Found {aggregation['total']} matching tickets."
            if aggregation["grouped_counts"]:
                groups = ", ".join(f"{k}: {v}" for k, v in aggregation["grouped_counts"].items())
                answer += f" Status breakdown: {groups}."
            results, tools, semantic_score = [], ["filter_tickets", "aggregate_tickets"], None
            evidence_state, evidence_reason = "sufficient", "Exact metadata aggregation was completed."
            lexical_support = None
            citations = validate_citations(answer, [])
        else:
            results, tools, semantic_score = self._retrieve(plan)
            evidence_state, evidence_reason, lexical_support = self._evidence_state(
                plan, results, semantic_score
            )
            if evidence_state == "insufficient":
                answer = "The retrieved Jira evidence is insufficient to answer this question reliably."
            elif plan.needs_generation:
                generation_question = plan.semantic_query
                if plan.route == "similarity":
                    generation_question = (
                        "NEW, UNVERIFIED REPORT:\n" + plan.semantic_query +
                        "\n\nCompare it with the historical Jira evidence. Describe every retrieved "
                        "source explicitly as a historical incident. Do not begin with 'the issue is "
                        "resolved' or 'the issue is unresolved'. Do not apply a historical status, root "
                        "cause or resolution to the new report. Begin the comparison with: "
                        "'A historical Jira incident documents...'"
                    )
                if any("\u0590" <= character <= "\u05ff" for character in plan.semantic_query):
                    generation_question += (
                        "\n\nAnswer in Hebrew because the user's question is in Hebrew. "
                        "Keep Jira ticket IDs unchanged."
                    )
                answer = self.generate_answer(generation_question, results)["answer"]
                if plan.route == "similarity":
                    answer = (
                        "New report:\nThe root cause and resolution have not yet been verified.\n\n"
                        "Historical incidents:\n" + answer +
                        "\n\nLimitation:\nThe cited history is an investigation lead only; it does not "
                        "verify the cause, priority, duplicate status or resolution of the new report."
                    )
                elif plan.route in {"semantic", "hybrid"} and self._is_abstention(answer):
                    evidence_state = "insufficient"
                    evidence_reason = (
                        "The generator inspected the retrieved context and explicitly found it "
                        "insufficient to support the requested claim."
                    )
            else:
                answer = self._deterministic_lookup(results[0])
            citations = validate_citations(answer, [row["document_id"] for row in results])

        solution_state_safe, solution_warnings = self._solution_state_safe(
            answer, results, citations["cited_ticket_ids"]
        )
        if not solution_state_safe:
            answer = (
                "The generated draft was withheld because its resolution wording conflicts with "
                "the cited ticket's documented solution state. Human review is required."
            )

        advisory_results = [
            row for row in results if row["document_id"] in citations["cited_ticket_ids"]
        ] or results[:1]
        advisory = self._advice(
            plan, advisory_results, evidence_state, new_ticket=new_ticket
        )
        if advisory["recommendation"]:
            answer = f"{answer}\n\nRecommendation: {advisory['recommendation']}"

        citation_safe = not citations["invalid_citations"]
        return {
            "question": text,
            "answer": answer,
            "plan": plan.as_dict(),
            "tools": tools,
            "evidence_state": evidence_state,
            "evidence_reason": evidence_reason,
            "semantic_score": semantic_score,
            "lexical_support": lexical_support,
            "source_ids": [row["document_id"] for row in results],
            "citation_safe": citation_safe,
            "invalid_citations": citations["invalid_citations"],
            "solution_state_safe": solution_state_safe,
            "solution_warnings": solution_warnings,
            "advice_present": advisory["recommendation"] is not None,
            "advisory": advisory,
            "write_executed": advisory["write_executed"],
        }


def evaluate_agent_scenarios(
    agent: AdvisoryAgent, scenarios: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run a small development harness without defining one exact tool trace per question."""
    rows = []
    for scenario in scenarios:
        result = agent.run(
            scenario["input"], new_ticket=scenario.get("new_ticket", False)
        )
        route_success = result["plan"]["route"] == scenario["expected_route"]
        evidence_safe = (
            result["evidence_state"] in scenario["allowed_evidence_states"]
            and result["solution_state_safe"]
        )
        advisory_safe = (
            result["write_executed"] is False
            and result["advice_present"] == scenario["expect_advice"]
        )
        passed = route_success and evidence_safe and result["citation_safe"] and advisory_safe
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "route": result["plan"]["route"],
            "tools": ", ".join(result["tools"]),
            "evidence": result["evidence_state"],
            "sources": ", ".join(result["source_ids"]),
            "route_success": route_success,
            "evidence_safe": evidence_safe,
            "citation_safe": result["citation_safe"],
            "advisory_safe": advisory_safe,
            "passed": passed,
            "result": result,
        })
    count = len(rows)
    metrics = {
        "route_success": sum(row["route_success"] for row in rows) / count,
        "evidence_safety": sum(row["evidence_safe"] for row in rows) / count,
        "citation_safety": sum(row["citation_safe"] for row in rows) / count,
        "advisory_safety": sum(row["advisory_safe"] for row in rows) / count,
        "end_to_end_success": sum(row["passed"] for row in rows) / count,
    }
    return rows, metrics


def ensure_agent_harness(
    path: Path,
    identity: dict[str, Any],
    build: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Load a compatible harness result or build it once after an input change."""
    path = Path(path)
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("identity") == identity and isinstance(saved.get("result"), dict):
                return saved["result"], "LOAD"
        except (OSError, json.JSONDecodeError):
            pass
    result = build()
    write_json_atomic(
        {"schema_version": "agent_harness_artifact_v1", "identity": identity, "result": result},
        path,
    )
    return result, "BUILD"
