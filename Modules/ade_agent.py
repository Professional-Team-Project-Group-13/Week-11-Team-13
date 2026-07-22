"""
 Agentic Orchestration Layer.

This module is what makes the system *agentic* rather than a fixed pipeline.
An autonomous agent receives a piece of text and decides, step by step, which
tools to invoke, inspects their outputs, and adapts its next action based on
what it observes — rather than executing a hard-coded sequence.

Design: a transparent ReAct-style loop (Reason -> Act -> Observe) over a
registry of tools. The agent is deliberately *interpretable* and *deterministic*
in its policy, which suits a clinical-safety setting where
every decision must be auditable. Each run produces a full trace of the agent's
reasoning, actions, and observations.

Tools the agent can call

    * classify        - run a model to get an ADE probability.
    * retrieve_cases  - find similar past cases via FAISS (case_retrieval).
    * explain         - generate LIME word importances (explainability).
    * assess_severity - derive a severity proxy from the text.
    * triage          - map probability + severity to a routing decision.

Autonomous behaviour 

    * If the model is confident and the retrieval-based second opinion agrees,
      the agent finalises without spending effort on explanation.
    * If the model is uncertain, the agent *chooses* to gather more evidence:
      it retrieves similar cases and generates an explanation before deciding.
    * If model and retrieved neighbours disagree, the agent escalates to a
      human — an autonomous safety action driven by observed conflict.

The agent's policy is decoupled from the tools, so swapping in a stronger model
or a different retriever does not change the orchestration logic.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import explainability as ex



# Tracing

@dataclass
class Step:
    """One Reason -> Act -> Observe cycle in the agent's trace."""

    thought: str
    action: str
    action_input: Any
    observation: Any


@dataclass
class AgentResult:
    """Final output of an agent run, including the full reasoning trace."""

    text: str
    ade_probability: float
    severity: str
    triage_level: str
    route: str
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    trace: List[Step] = field(default_factory=list)

    def summary(self):
        """One-line human-readable summary of the decision."""
        return (f"ADE p={self.ade_probability:.2f} | "
                f"severity={self.severity} | {self.triage_level} "
                f"-> {self.route}")



# Tool registry

class ToolRegistry:
    """Holds the callable tools the agent may invoke, by name."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name, fn):
        self._tools[name] = fn
        return self

    def has(self, name):
        return name in self._tools

    def call(self, name, *args, **kwargs):
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name](*args, **kwargs)



# The agent

class ADEAgent:
    """
    An autonomous agent for adverse-drug-event triage.

    Parameters
    ----------
    tools : ToolRegistry
        Must provide at least 'classify'. May provide 'retrieve_cases',
        'explain', 'assess_severity', 'triage'.
    confidence_margin : float
        Distance from 0.5 within which the model is considered "uncertain",
        triggering additional evidence-gathering.
    """

    def __init__(self, tools, confidence_margin=0.15):
        self.tools = tools
        self.confidence_margin = confidence_margin

    #  individual reasoning helpers 
    def _is_uncertain(self, proba):
        return abs(proba - 0.5) < self.confidence_margin

    def run(self, text, k_cases=5):
        """
        Execute the agent loop on one input and return an AgentResult.

        The control flow is genuinely conditional: which tools run, and in what
        order, depends on what the agent observes at each step.
        """
        trace: List[Step] = []
        evidence: Dict[str, Any] = {}

        # Step 1: always classify first 
        proba = float(self.tools.call("classify", text))
        trace.append(Step(
            thought="I must first estimate ADE probability with the model.",
            action="classify", action_input=text, observation=proba,
        ))

        #  Step 2: decide whether more evidence is needed 
        uncertain = self._is_uncertain(proba)
        retrieval_pred, retrieval_conf = None, None
        if uncertain and self.tools.has("retrieve_cases"):
            thought = ("Model is uncertain (probability near 0.5); I will "
                       "gather a second opinion from similar past cases.")
            vote = self.tools.call("retrieve_cases", text, k_cases)
            retrieval_pred = vote.get("pred")
            retrieval_conf = vote.get("confidence")
            evidence["retrieved_cases"] = vote.get("neighbours", [])
            evidence["retrieval_vote"] = {
                "pred": retrieval_pred, "confidence": retrieval_conf,
            }
            trace.append(Step(
                thought=thought, action="retrieve_cases",
                action_input={"text": text, "k": k_cases}, observation=vote,
            ))
        else:
            trace.append(Step(
                thought=("Model is confident; no extra retrieval needed for "
                         "the decision."),
                action="skip_retrieval", action_input=None,
                observation="confident",
            ))

        #  Step 3: detect model/retrieval conflict 
        model_pred = 1 if proba >= 0.5 else 0
        conflict = (retrieval_pred is not None
                    and retrieval_pred != model_pred
                    and (retrieval_conf or 0) >= 0.6)
        if conflict:
            trace.append(Step(
                thought=("Model and retrieved cases disagree with reasonable "
                         "confidence; safest action is human escalation."),
                action="flag_conflict", action_input=None, observation=True,
            ))

        #  Step 4: severity assessment 
        if self.tools.has("assess_severity"):
            severity = self.tools.call("assess_severity", text)
        else:
            severity = "mild"
        trace.append(Step(
            thought="Assess severity to inform triage routing.",
            action="assess_severity", action_input=text, observation=severity,
        ))

        # --- Step 5: explain when it adds value 
        # Per the explainability design, LIME and SHAP run together at this
        # single explanation step (after routing), so the human reviewer sees
        # a feature-based explanation from two complementary methods. The agent
        # explains for positive or uncertain predictions, where justification
        # matters most.
        if (uncertain or model_pred == 1) and (
                self.tools.has("explain") or self.tools.has("explain_shap")):
            explanations = {}
            if self.tools.has("explain"):
                try:
                    explanations["lime_top_words"] = self.tools.call(
                        "explain", text)
                except Exception as exc:
                    explanations["lime_error"] = str(exc)
            if self.tools.has("explain_shap"):
                try:
                    explanations["shap_top_words"] = self.tools.call(
                        "explain_shap", text)
                except Exception as exc:
                    explanations["shap_error"] = str(exc)
            evidence.update(explanations)
            # Note whether the two methods agree on the top drivers — agreement
            # is evidence the model uses clinically meaningful signal.
            agree = _explanations_agree(
                explanations.get("lime_top_words"),
                explanations.get("shap_top_words"))
            if agree is not None:
                evidence["lime_shap_agreement"] = agree
            trace.append(Step(
                thought=("Prediction is positive or uncertain; running LIME and "
                         "SHAP together to justify it for the human reviewer."),
                action="explain (LIME + SHAP)", action_input=text,
                observation=explanations,
            ))

        #  Step 6: triage decision 
        if self.tools.has("triage"):
            decision = self.tools.call("triage", proba, severity)
        else:
            decision = ex.triage(proba, severity)

        # The agent can override the tool's routing on observed conflict:
        if conflict:
            decision["route"] = "human_review"
            decision["rationale"] = (
                "Escalated: model and similar-case evidence disagree. "
                + decision.get("rationale", "")
            )
        trace.append(Step(
            thought="Combine probability and severity into a routing decision.",
            action="triage",
            action_input={"proba": proba, "severity": severity},
            observation=decision,
        ))

        return AgentResult(
            text=text,
            ade_probability=round(proba, 3),
            severity=severity,
            triage_level=decision["triage_level"],
            route=decision["route"],
            rationale=decision["rationale"],
            evidence=evidence,
            trace=trace,
        )

    def run_batch(self, texts, k_cases=5):
        """Run the agent over many texts; returns a list of AgentResult."""
        return [self.run(t, k_cases=k_cases) for t in texts]



# Convenience builder

def build_agent(classify_fn, retriever=None, explain_fn=None, shap_fn=None,
                severity_fn=None, triage_fn=None, confidence_margin=0.15):
    """
    Wire common components into a ready-to-use ADEAgent.

    """
    tools = ToolRegistry()
    tools.register("classify", classify_fn)

    if retriever is not None:
        def _retrieve(text, k):
            neighbours = retriever.query(text, k=k)
            pred, conf = retriever.retrieval_vote(text, k=k)
            return {"neighbours": neighbours, "pred": pred, "confidence": conf}
        tools.register("retrieve_cases", _retrieve)

    if explain_fn is not None:
        tools.register("explain", explain_fn)
    if shap_fn is not None:
        tools.register("explain_shap", shap_fn)
    if severity_fn is not None:
        tools.register("assess_severity", severity_fn)
    if triage_fn is not None:
        tools.register("triage", triage_fn)

    return ADEAgent(tools, confidence_margin=confidence_margin)


def format_trace(result):
    """Render an AgentResult's reasoning trace as readable text."""
    lines = [f"AGENT DECISION: {result.summary()}",
             f"Rationale: {result.rationale}", "", "Reasoning trace:"]
    for i, step in enumerate(result.trace, start=1):
        obs = step.observation
        if isinstance(obs, (dict, list)):
            obs = str(obs)[:120]
        lines.append(f"  [{i}] THINK: {step.thought}")
        lines.append(f"      ACT:   {step.action}({_short(step.action_input)})")
        lines.append(f"      OBS:   {obs}")
    return "\n".join(lines)


def _short(value):
    text = str(value)
    return text[:50] + "..." if len(text) > 50 else text


def _explanations_agree(lime_words, shap_words, top_n=5):
    """
    Report whether LIME and SHAP agree on the most influential words.

    Returns the overlap fraction among each method's top-n words, or None if
    either explanation is unavailable. Agreement is evidence that the model is
    keying on the same signal under two different attribution methods.
    """
    if not lime_words or not shap_words:
        return None
    lime_top = {str(w).lower() for w, _ in lime_words[:top_n]}
    shap_top = {str(w).lower() for w, _ in shap_words[:top_n]}
    if not lime_top or not shap_top:
        return None
    overlap = len(lime_top & shap_top) / len(lime_top | shap_top)
    return round(overlap, 3)
