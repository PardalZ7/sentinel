# Copyright 2026 Alexandre Cardoso
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Denial rules evaluation use case.

Evaluates deterministic business-rule conditions against an event_dict before
ML detectors run. If any rule's condition evaluates to True, the event is
immediately flagged as anomalous (AnomalyType.RULE) and detectors are bypassed.

Field resolution in conditions:
  - "input.field"  → event_dict["_input_body"]["field"]
  - "output.field" → event_dict["_output_body"]["field"]
  - "field"        → event_dict["field"]  (operational fields like processing_latency_ms)

Special function:
  - required(path): True if the field exists and is not None.

Missing fields in comparisons evaluate silently as absent (rule does not fire).
Malformed conditions log a warning and do not fire (never block the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simpleeval import EvalWithCompoundTypes, NameNotDefined, InvalidExpression

from sentinel.config.schema import DenialRuleConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_SENTINEL = object()


@dataclass
class DenialRuleHit:
    rule_name: str
    score: float


def _resolve_field(path: str, event_dict: dict) -> Any:
    """Resolve a dot-notation field path against event_dict.

    Prefixes:
      "input.*"  → _input_body
      "output.*" → _output_body
      anything else → top-level event_dict key
    """
    if path.startswith("input."):
        body = event_dict.get("_input_body") or {}
        key = path[len("input."):]
        return body.get(key, _SENTINEL)
    if path.startswith("output."):
        body = event_dict.get("_output_body") or {}
        key = path[len("output."):]
        return body.get(key, _SENTINEL)
    return event_dict.get(path, _SENTINEL)


def _build_names(event_dict: dict) -> dict[str, Any]:
    """Build the flat names dict that simpleeval uses to resolve identifiers.

    Identifiers in conditions use underscore-separated names internally
    (simpleeval doesn't support dot-access natively). We expose the raw
    dicts so that users can write "input.field" — simpleeval treats "input"
    as a dict lookup and "field" as a key access when using EvalWithCompoundTypes.
    """
    names: dict[str, Any] = {}

    input_body = event_dict.get("_input_body")
    output_body = event_dict.get("_output_body")

    # Expose "input" and "output" as dicts for dot-access in expressions.
    # If multi-input, _input_body is already a {name: body} dict.
    names["input"] = input_body if isinstance(input_body, dict) else {}
    names["output"] = output_body if isinstance(output_body, dict) else {}

    # Expose top-level operational fields
    for k, v in event_dict.items():
        if not k.startswith("_"):
            names[k] = v

    return names


def _make_required_fn(event_dict: dict):
    """Return a required() callable for use in denial rule conditions.

    required(path) returns True when the field is ABSENT or None — meaning the
    "required field" constraint is VIOLATED. Since a denial rule fires when its
    condition is True, this encodes: "anomaly when this field is missing".

    Examples in conditions:
      required('output.result')               → anomaly if result is absent/None
      required('output.result') or required('output.error')
                                              → anomaly if BOTH are absent/None
    """
    def required(path: str) -> bool:
        val = _resolve_field(path, event_dict)
        return val is _SENTINEL or val is None
    return required


def evaluate_denial_rules(
    rules: list[DenialRuleConfig],
    event_dict: dict,
) -> DenialRuleHit | None:
    """Evaluate all denial rules against event_dict, return first hit or None.

    Rules are evaluated in declaration order. Short-circuits on first match.
    """
    if not rules:
        return None

    names = _build_names(event_dict)
    required_fn = _make_required_fn(event_dict)

    for rule in rules:
        try:
            evaluator = EvalWithCompoundTypes(
                names=names,
                functions={"required": required_fn},
            )
            result = evaluator.eval(rule.condition)
            if result is True:
                logger.info(
                    "denial_rule_hit",
                    rule=rule.name,
                    condition=rule.condition,
                    score=rule.score,
                )
                return DenialRuleHit(rule_name=rule.name, score=rule.score)
        except NameNotDefined as exc:
            logger.warning(
                "denial_rule_field_missing",
                rule=rule.name,
                error=str(exc),
            )
        except (InvalidExpression, Exception) as exc:
            logger.warning(
                "denial_rule_eval_error",
                rule=rule.name,
                condition=rule.condition,
                error=str(exc),
            )

    return None
