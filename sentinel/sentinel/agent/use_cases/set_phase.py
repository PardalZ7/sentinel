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

"""Phase transition use case for agents and individual detectors.

All functions are pure — they receive state and return new state.
Side effects (e.g. forcing send_all_events=True) are the caller's responsibility.
"""

from dataclasses import replace

from sentinel.domain.models import ModelPhase
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def set_phase(current_state, new_phase: ModelPhase, detector_name: str | None = None):
    """Return a new agent state with the updated phase.

    Args:
        current_state:   AgentState instance.
        new_phase:       Target ModelPhase.
        detector_name:   If provided, only that detector's phase changes.
                         If None, all detectors are updated (backward-compat).

    Returns a new AgentState (dataclasses.replace semantics).
    """
    agent_name = getattr(current_state, "agent_name", "unknown")
    new_detectors = dict(current_state.detectors)

    target_names = [detector_name] if detector_name else list(new_detectors.keys())

    for dname in target_names:
        if dname not in new_detectors:
            logger.warning(
                "phase_transition_detector_not_found",
                agent=agent_name,
                detector=dname,
            )
            continue
        det = new_detectors[dname]
        old_phase = det.phase
        new_detectors[dname] = replace(det, phase=new_phase)
        logger.info(
            "phase_transition",
            agent=agent_name,
            detector=dname,
            from_phase=old_phase,
            to_phase=new_phase,
        )

    return replace(current_state, detectors=new_detectors)
