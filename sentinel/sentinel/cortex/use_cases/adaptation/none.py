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

"""No-adaptation strategy — model and baseline_error are fixed after initial training."""

import numpy as np

from sentinel.cortex.use_cases.adaptation.base import AdaptationContext, IAdaptationStrategy


class NoAdaptationStrategy(IAdaptationStrategy):
    """Passes every inference sample through without modifying model or threshold.

    Appropriate for fully static environments where any drift is itself a signal.
    """

    @property
    def name(self) -> str:
        return "none"

    def on_inference_sample(
        self,
        sample: np.ndarray,
        ctx: AdaptationContext,
    ) -> AdaptationContext:
        return ctx

    def on_phase_entered(self, ctx: AdaptationContext) -> AdaptationContext:
        return ctx
