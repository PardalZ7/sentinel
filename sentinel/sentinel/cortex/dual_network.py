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

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from sentinel.cortex.use_cases.autoencoder import (
    SentinelAutoencoder,
    compute_reconstruction_error,
    is_systemic_anomaly,
)
from sentinel.cortex.use_cases.causal_chain import CausalDiagnosis, analyze_causality
from sentinel.domain.models import Alert, AlertSeverity, AlertType
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DualNetworkResult:
    causal: CausalDiagnosis | None
    systemic_error: float
    is_systemic: bool
    final_alert: Alert | None


def run_dual_network(
    causal_window: deque,
    autoencoder_model: SentinelAutoencoder | None,
    state_vectors: dict,
    config: object,  # CortexConfig
    baseline_error: float = 0.01,
    source_name: str = "cortex",
) -> DualNetworkResult:
    """Run both Causal Chain and Autoencoder analyses and fuse results into an Alert."""
    # 1. Causal chain analysis
    causal_diagnosis: CausalDiagnosis | None = None
    try:
        causal_diagnosis = analyze_causality(causal_window)
    except Exception as exc:
        logger.warning("causal_chain_failed", error=str(exc))

    # 2. Autoencoder / systemic anomaly analysis
    systemic_error = 0.0
    detected_systemic = False

    if autoencoder_model is not None and state_vectors:
        try:
            from sentinel.cortex.use_cases.aggregate import build_feature_matrix

            matrix = build_feature_matrix(state_vectors)
            if matrix.shape[0] > 0:
                # Use mean of all agent feature vectors as system state
                mean_features = matrix.mean(axis=0)
                systemic_error = compute_reconstruction_error(
                    autoencoder_model, mean_features
                )
                detected_systemic = is_systemic_anomaly(systemic_error, baseline_error)
        except Exception as exc:
            logger.warning("autoencoder_failed", error=str(exc))

    # 3. Fuse into Alert
    final_alert: Alert | None = None

    has_causal = (
        causal_diagnosis is not None
        and causal_diagnosis.root_agent is not None
        and len(causal_diagnosis.sequence) > 1
    )

    if has_causal and detected_systemic:
        severity = AlertSeverity.CRITICAL
        alert_type = AlertType.SYSTEMIC
        affected = list(
            set(causal_diagnosis.sequence + list(state_vectors.keys()))
        )
        description = (
            f"Systemic anomaly detected (error={systemic_error:.4f}) with causal chain "
            f"originating at {causal_diagnosis.root_agent}"
        )
        diagnosis = {
            "causal": {
                "root_agent": causal_diagnosis.root_agent,
                "sequence": causal_diagnosis.sequence,
                "confidence": causal_diagnosis.confidence,
            },
            "systemic_error": systemic_error,
        }
    elif has_causal:
        severity = AlertSeverity.WARNING
        alert_type = AlertType.CAUSAL
        affected = causal_diagnosis.sequence
        description = (
            f"Causal anomaly chain detected: root={causal_diagnosis.root_agent}, "
            f"affected={causal_diagnosis.affected_agents}"
        )
        diagnosis = {
            "causal": {
                "root_agent": causal_diagnosis.root_agent,
                "sequence": causal_diagnosis.sequence,
                "confidence": causal_diagnosis.confidence,
            }
        }
    elif detected_systemic:
        severity = AlertSeverity.WARNING
        alert_type = AlertType.SYSTEMIC
        affected = list(state_vectors.keys())
        description = f"Systemic anomaly detected (reconstruction_error={systemic_error:.4f})"
        diagnosis = {"systemic_error": systemic_error}
    else:
        return DualNetworkResult(
            causal=causal_diagnosis,
            systemic_error=systemic_error,
            is_systemic=detected_systemic,
            final_alert=None,
        )

    final_alert = Alert(
        source=source_name,
        severity=severity,
        alert_type=alert_type,
        affected_agents=affected,
        description=description,
        timestamp=datetime.now(tz=timezone.utc),
        diagnosis=diagnosis,
    )

    return DualNetworkResult(
        causal=causal_diagnosis,
        systemic_error=systemic_error,
        is_systemic=detected_systemic,
        final_alert=final_alert,
    )
