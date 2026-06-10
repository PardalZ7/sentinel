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

"""Factory for creating IDetector instances from DetectorConfig.

Centralises the algorithm → class mapping so new algorithms can be added
by registering one entry here without touching any other module.
"""

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector


def create_detector(config: DetectorConfig) -> IDetector:
    """Instantiate the appropriate IDetector for the given config.

    Raises:
        ValueError: if config.algorithm is not a known algorithm identifier.
    """
    match config.algorithm:
        case "isolation_forest":
            from sentinel.agent.detectors.isolation_forest_detector import IsolationForestDetector
            return IsolationForestDetector(config)

        case "cvae":
            from sentinel.agent.detectors.cvae_detector import ConditionalVAEDetector
            return ConditionalVAEDetector(config)

        case "maf":
            from sentinel.agent.detectors.maf_detector import MAFDetector
            return MAFDetector(config)

        case "nri":
            from sentinel.agent.detectors.nri_detector import NRIDetector
            return NRIDetector(config)

        case "vae":
            from sentinel.agent.detectors.vae_detector import VAEDetector
            return VAEDetector(config)

        case "svdd":
            from sentinel.agent.detectors.svdd_detector import SVDDDetector
            return SVDDDetector(config)

        case "lstm_ae":
            from sentinel.agent.detectors.lstm_ae_detector import LSTMAEDetector
            return LSTMAEDetector(config)

        case "tcn_ae":
            from sentinel.agent.detectors.tcn_ae_detector import TCNAEDetector
            return TCNAEDetector(config)

        case _:
            raise ValueError(
                f"Unknown detector algorithm '{config.algorithm}'. "
                f"Valid options: isolation_forest, cvae, maf, nri, vae, svdd, lstm_ae, tcn_ae."
            )
