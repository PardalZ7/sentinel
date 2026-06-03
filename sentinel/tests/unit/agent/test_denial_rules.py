import pytest

from sentinel.agent.use_cases.denial_rules import DenialRuleHit, evaluate_denial_rules
from sentinel.config.schema import DenialRuleConfig


def _event(input_body=None, output_body=None, **kwargs):
    base = {
        "processing_latency_ms": 100.0,
        "input_size_bytes": 512,
        "output_size_bytes": 256,
        "payload_schema_hash": "abc123",
        "_input_body": input_body or {},
        "_output_body": output_body or {},
    }
    base.update(kwargs)
    return base


def _rule(condition: str, name: str = "test_rule", score: float = -1.0) -> DenialRuleConfig:
    return DenialRuleConfig(name=name, condition=condition, score=score)


# ── required() — semântica: fires quando campo AUSENTE ou None ────────────────

def test_required_field_absent_fires():
    """required(field) → anomalia quando campo está ausente."""
    event = _event(output_body={})
    hit = evaluate_denial_rules([_rule("required('output.result')")], event)
    assert hit is not None
    assert hit.rule_name == "test_rule"
    assert hit.score == -1.0


def test_required_field_none_fires():
    """required(field) → anomalia quando campo é None."""
    event = _event(output_body={"result": None})
    hit = evaluate_denial_rules([_rule("required('output.result')")], event)
    assert hit is not None


def test_required_field_present_does_not_fire():
    """required(field) → não dispara quando campo existe e não é None."""
    event = _event(output_body={"result": "ok"})
    hit = evaluate_denial_rules([_rule("required('output.result')")], event)
    assert hit is None


def test_required_input_field_absent_fires():
    event = _event(input_body={})
    hit = evaluate_denial_rules([_rule("required('input.amount')")], event)
    assert hit is not None


def test_required_input_field_present_does_not_fire():
    event = _event(input_body={"amount": 10})
    hit = evaluate_denial_rules([_rule("required('input.amount')")], event)
    assert hit is None


def test_required_or_both_absent_fires():
    """required(a) or required(b) → dispara se QUALQUER UM estiver ausente."""
    event = _event(output_body={})
    hit = evaluate_denial_rules([_rule("required('output.result') or required('output.error')")], event)
    assert hit is not None  # True or True = True


def test_required_or_one_present_still_fires():
    """required(a) or required(b) com apenas 'a' ausente → ainda dispara (True or False)."""
    event = _event(output_body={"error": "oops"})
    hit = evaluate_denial_rules([_rule("required('output.result') or required('output.error')")], event)
    assert hit is not None  # required(result)=True or required(error)=False → True


def test_required_or_both_present_does_not_fire():
    """required(a) or required(b) → não dispara quando ambos estão presentes."""
    event = _event(output_body={"result": "ok", "error": None})
    # result presente → required=False; error é None → required=True → True or False... hmm
    # Correto: output.error = None → required = True (ausente/None conta como violação)
    # Então: False or True = True → fires
    # Para não disparar, ambos devem estar presentes E não-None
    event2 = _event(output_body={"result": "ok", "error": "nope"})
    hit = evaluate_denial_rules([_rule("required('output.result') or required('output.error')")], event2)
    assert hit is None  # False or False = False


def test_required_and_both_absent_fires():
    """required(a) and required(b) → dispara apenas quando AMBOS estão ausentes."""
    event = _event(output_body={})
    hit = evaluate_denial_rules([_rule("required('output.result') and required('output.error')")], event)
    assert hit is not None  # True and True = True


def test_required_and_one_present_does_not_fire():
    """required(a) and required(b) com 'a' presente → não dispara (False and True = False)."""
    event = _event(output_body={"result": "ok"})
    hit = evaluate_denial_rules([_rule("required('output.result') and required('output.error')")], event)
    assert hit is None  # required(result)=False and required(error)=True → False


# ── comparações simples ───────────────────────────────────────────────────────

def test_equality_match_fires():
    event = _event(output_body={"status": "error"})
    hit = evaluate_denial_rules([_rule("output['status'] == 'error'")], event)
    assert hit is not None


def test_equality_no_match_does_not_fire():
    event = _event(output_body={"status": "success"})
    hit = evaluate_denial_rules([_rule("output['status'] == 'error'")], event)
    assert hit is None


def test_not_equal_fires():
    event = _event(output_body={"code": 500})
    hit = evaluate_denial_rules([_rule("output['code'] != 200")], event)
    assert hit is not None


def test_greater_than_fires():
    event = _event(output_body={"latency": 5000})
    hit = evaluate_denial_rules([_rule("output['latency'] > 1000")], event)
    assert hit is not None


def test_less_than_fires():
    event = _event(output_body={"score": 0.1})
    hit = evaluate_denial_rules([_rule("output['score'] < 0.5")], event)
    assert hit is not None


def test_and_both_true_fires():
    event = _event(output_body={"status": "success", "result": None})
    hit = evaluate_denial_rules(
        [_rule("output['status'] == 'success' and output['result'] == None")], event
    )
    assert hit is not None


def test_and_one_false_does_not_fire():
    event = _event(output_body={"status": "error", "result": None})
    hit = evaluate_denial_rules(
        [_rule("output['status'] == 'success' and output['result'] == None")], event
    )
    assert hit is None


# ── campo ausente em comparação normal ───────────────────────────────────────

def test_missing_field_in_comparison_does_not_fire():
    """Campo ausente em comparação normal não dispara (silencioso)."""
    event = _event(output_body={})
    hit = evaluate_denial_rules([_rule("output['status'] == 'success'")], event)
    assert hit is None


# ── score customizado ─────────────────────────────────────────────────────────

def test_custom_score_propagated():
    event = _event(output_body={})
    hit = evaluate_denial_rules([_rule("required('output.result')", score=-0.5)], event)
    assert hit is not None
    assert hit.score == -0.5


# ── short-circuit (primeiro hit) ─────────────────────────────────────────────

def test_first_matching_rule_returned():
    rules = [
        _rule("required('output.x')", name="rule_a", score=-0.9),
        _rule("required('output.y')", name="rule_b", score=-0.5),
    ]
    event = _event(output_body={})
    hit = evaluate_denial_rules(rules, event)
    assert hit is not None
    assert hit.rule_name == "rule_a"


def test_second_rule_fires_when_first_does_not():
    rules = [
        _rule("output['status'] == 'error'", name="rule_a"),
        _rule("required('output.result')", name="rule_b"),
    ]
    event = _event(output_body={"status": "success"})
    hit = evaluate_denial_rules(rules, event)
    assert hit is not None
    assert hit.rule_name == "rule_b"


# ── lista vazia e backward compat ────────────────────────────────────────────

def test_empty_rules_returns_none():
    event = _event(output_body={"result": None})
    assert evaluate_denial_rules([], event) is None


# ── expressão inválida não bloqueia pipeline ──────────────────────────────────

def test_invalid_condition_does_not_raise():
    event = _event(output_body={})
    hit = evaluate_denial_rules([_rule("this is not valid python !!!")], event)
    assert hit is None


# ── campos operacionais (top-level) ──────────────────────────────────────────

def test_operational_field_comparison():
    event = _event(output_body={}, processing_latency_ms=5000.0)
    hit = evaluate_denial_rules([_rule("processing_latency_ms > 3000")], event)
    assert hit is not None
