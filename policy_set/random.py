from typing import Any, TypeVar, cast

import numpy as np

from data import Batch
from data.batch import BatchProtocol
from data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from policy_set import BasePolicy
from policy_set.base import TrainingStats


class RandomTrainingStats(TrainingStats):
    pass


TRandomTrainingStats = TypeVar("TRandomTrainingStats", bound=RandomTrainingStats)


class RandomPolicy(BasePolicy[TRandomTrainingStats]):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute the random action over the given batch data.
        """
        mask = batch.obs.mask  # type: ignore
        logits = np.random.rand(*mask.shape)
        logits[~mask] = -np.inf
        result = Batch(act=logits.argmax(axis=-1))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TRandomTrainingStats:  # type: ignore
        """Since a random agent learns nothing, it returns an empty dict."""
        return RandomTrainingStats()  # type: ignore[return-value]
