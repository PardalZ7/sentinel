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

"""Test-buffer evaluation use case.

Scores every sample in the test buffer against the current champion model,
counting samples incorrectly flagged as anomalies (false positives).

Does NOT change detector phase.
"""

from sentinel.domain.ports.detector import DetectorState, IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def run_detector_test(detector: IDetector, det_state: DetectorState) -> dict:
    """Score all samples in the test buffer against the current champion model.

    Returns a dict with:
      samples_run  — number of test samples evaluated
      failed       — samples incorrectly flagged as anomalous (false positives)
      fp_rate      — failed / samples_run  (0.0 if no samples)

    A sample is 'failed' if the model flags it as anomalous — indicating the
    model incorrectly classifies normal traffic (high false-positive rate).
    """
    if det_state.model is None:
        return {"error": "no_model", "samples_run": 0, "failed": 0, "fp_rate": 0.0}

    buffer = det_state.test_buffer
    if not buffer:
        return {"samples_run": 0, "failed": 0, "fp_rate": 0.0}

    failed = 0
    for sample in buffer:
        try:
            score = detector.score(det_state.model, sample)
            if detector.is_anomaly(score):
                failed += 1
        except Exception as exc:
            logger.warning(
                "test_sample_scoring_failed",
                detector=detector.name,
                error=str(exc),
            )
            failed += 1

    fp_rate = failed / len(buffer)
    result = {"samples_run": len(buffer), "failed": failed, "fp_rate": round(fp_rate, 4)}

    logger.info(
        "detector_test_complete",
        detector=detector.name,
        **result,
    )
    return result
