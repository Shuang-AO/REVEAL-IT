import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from overrides import override
from torch import nn

from data import ReplayBuffer, SequenceSummaryStats, to_numpy, to_torch_as
from data.batch import Batch, BatchProtocol, TArr
from data.buffer.base import TBuffer
from data.types import (
    ActBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from utils import MultipleLRSchedulers
from utils.print import DataclassPPrintMixin

logger = logging.getLogger(__name__)

TLearningRateScheduler: TypeAlias = torch.optim.lr_scheduler.LRScheduler | MultipleLRSchedulers


@dataclass(kw_only=True)
class TrainingStats(DataclassPPrintMixin):
    _non_loss_fields = ("train_time", "smoothed_loss")

    train_time: float = 0.0
    """The time for learning models."""

    # TODO: modified in the trainer but not used anywhere else. Should be refactored.
    smoothed_loss: dict = field(default_factory=dict)
    """The smoothed loss statistics of the policy learn step."""

    # Mainly so that we can override this in the TrainingStatsWrapper
    def _get_self_dict(self) -> dict[str, Any]:
        return self.__dict__

    def get_loss_stats_dict(self) -> dict[str, float]:
        result = {}
        for k, v in self._get_self_dict().items():
            if k.startswith("_"):
                logger.debug(f"Skipping {k=} as it starts with an underscore.")
                continue
            if k in self._non_loss_fields or v is None:
                continue
            if isinstance(v, SequenceSummaryStats):
                result[k] = v.mean
            else:
                result[k] = v

        return result


class TrainingStatsWrapper(TrainingStats):
    _setattr_frozen = False
    _training_stats_public_fields = TrainingStats.__dataclass_fields__.keys()

    def __init__(self, wrapped_stats: TrainingStats) -> None:
        """In this particular case, super().__init__() should be called LAST in the subclass init."""
        self._wrapped_stats = wrapped_stats

        # HACK: special sauce for the existing attributes of the base TrainingStats class
        # for some reason, delattr doesn't work here, so we need to delegate their handling
        # to the wrapped stats object by always keeping the value there and in self in sync
        # see also __setattr__
        for k in self._training_stats_public_fields:
            super().__setattr__(k, getattr(self._wrapped_stats, k))

        self._setattr_frozen = True

    @override
    def _get_self_dict(self) -> dict[str, Any]:
        return {**self._wrapped_stats._get_self_dict(), **self.__dict__}

    @property
    def wrapped_stats(self) -> TrainingStats:
        return self._wrapped_stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_stats, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # HACK: special sauce for the existing attributes of the base TrainingStats class, see init
        # Need to keep them in sync with the wrapped stats object
        if name in self._training_stats_public_fields:
            setattr(self._wrapped_stats, name, value)
            super().__setattr__(name, value)
            return

        if not self._setattr_frozen:
            super().__setattr__(name, value)
            return

        if not hasattr(self, name):
            raise AttributeError(
                f"Setting new attributes on StatsWrappers outside of init is not allowed. "
                f"Tried to set {name=}, {value=} on {self.__class__.__name__}. \n"
                f"NOTE: you may get this error if you call super().__init__() in your subclass init too early! "
                f"The call to super().__init__() should be the last call in your subclass init.",
            )
        if hasattr(self._wrapped_stats, name):
            setattr(self._wrapped_stats, name, value)
        else:
            super().__setattr__(name, value)


TTrainingStats = TypeVar("TTrainingStats", bound=TrainingStats)


class BasePolicy(nn.Module, Generic[TTrainingStats], ABC):
    def __init__(
        self,
        *,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        allowed_action_bound_methods = ("clip", "tanh")
        if (
            action_bound_method is not None
            and action_bound_method not in allowed_action_bound_methods
        ):
            raise ValueError(
                f"Got invalid {action_bound_method=}. "
                f"Valid values are: {allowed_action_bound_methods}.",
            )
        if action_scaling and not isinstance(action_space, Box):
            raise ValueError(
                f"action_scaling can only be True when action_space is Box but "
                f"got: {action_space}",
            )

        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        if isinstance(action_space, Discrete | MultiDiscrete | MultiBinary):
            self.action_type = "discrete"
        elif isinstance(action_space, Box):
            self.action_type = "continuous"
        else:
            raise ValueError(f"Unsupported action space: {action_space}.")
        self.agent_id = 0
        self.updating = False
        self.action_scaling = action_scaling
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler
        self._compile()

    def set_agent_id(self, agent_id: int) -> None:
        """Set self.agent_id = agent_id, for MARL."""
        self.agent_id = agent_id

    def exploration_noise(
        self,
        act: np.ndarray | BatchProtocol,
        batch: RolloutBatchProtocol,
    ) -> np.ndarray | BatchProtocol:
        return act

    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters(), strict=True):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def compute_action(
        self,
        obs: TArr,
        info: dict[str, Any] | None = None,
        state: dict | BatchProtocol | np.ndarray | None = None,
    ) -> np.ndarray | int:

        # need to add empty batch dimension
        obs = obs[None, :]
        obs_batch = cast(ObsBatchProtocol, Batch(obs=obs, info=info))
        act = self.forward(obs_batch, state=state).act.squeeze()
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        act = self.map_action(act)
        if isinstance(self.action_space, Discrete):
            # could be an array of shape (), easier to just convert to int
            act = int(act)  # type: ignore
        return act

    @abstractmethod
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:"""Compute action over the given batch data.
    """
   
    @staticmethod
    def _action_to_numpy(act: TArr) -> np.ndarray:
        act = to_numpy(act)  # NOTE: to_numpy could confusingly also return a Batch
        if not isinstance(act, np.ndarray):
            raise ValueError(
                f"act should have been be a numpy.ndarray, but got {type(act)}.",
            )
        return act

    def map_action(
        self,
        act: TArr,
    ) -> np.ndarray:
      
        act = self._action_to_numpy(act)
        if isinstance(self.action_space, gym.spaces.Box):
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert (
                    np.min(act) >= -1.0 and np.max(act) <= 1.0
                ), f"action scaling only accepts raw action range = [-1, 1], but got: {act}"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0
        return act

    def map_action_inverse(
        self,
        act: TArr,
    ) -> np.ndarray:
        act = self._action_to_numpy(act)
        if isinstance(self.action_space, gym.spaces.Box):
            if self.action_scaling:
                low, high = self.action_space.low, self.action_space.high
                scale = high - low
                eps = np.finfo(np.float32).eps.item()
                scale[scale < eps] += eps
                act = (act - low) * 2.0 / scale - 1.0
            if self.action_bound_method == "tanh":
                act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0

        return act

    def process_buffer(self, buffer: TBuffer) -> TBuffer:
        return buffer

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        return batch

    @abstractmethod
    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        """Update policy with a given batch of data.

        :return: A dataclass object, including the data needed to be logged (e.g., loss).
            If you use ``torch.distributions.Normal`` and
            ``torch.distributions.Categorical`` to calculate the log_prob,
            please be careful about the shape: Categorical distribution gives
            "[batch_size]" shape while Normal distribution gives "[batch_size,
            1]" shape. The auto-broadcasting of numerical operation with torch
            tensors will amplify this error.
        """
    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
      
        if hasattr(buffer, "update_weight"):
            if hasattr(batch, "weight"):
                buffer.update_weight(indices, batch.weight)
            else:
                logger.warning(
                    "batch has no attribute 'weight', but buffer has an "
                    "update_weight method. This is probably a mistake."
                    "Prioritized replay is disabled for this batch.",
                )

    def update(
        self,
        sample_size: int | None,
        buffer: ReplayBuffer | None,
        **kwargs: Any,
    ) -> TTrainingStats:
        # TODO: when does this happen?
        # -> this happens never in practice as update is either called with a collector buffer or an assert before
        if buffer is None:
            return TrainingStats()  # type: ignore[return-value]
        start_time = time.time()
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        training_stat = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        training_stat.train_time = time.time() - start_time
        return training_stat

    @staticmethod
    def value_mask(buffer: ReplayBuffer, indices: np.ndarray) -> np.ndarray:
        return ~buffer.terminated[indices]

    @staticmethod
    def compute_episodic_return(
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: np.ndarray | torch.Tensor | None = None,
        v_s: np.ndarray | torch.Tensor | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute returns over given batch.
        """
        rew = batch.rew
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * BasePolicy.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage

    @staticmethod
    def compute_nstep_return(
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> BatchWithReturnsProtocol:
        r"""Compute n-step return for Q-learning targets.
        """
        assert not rew_norm, "Reward normalization in computing n-step returns is unsupported now."
        if len(indices) != len(batch):
            raise ValueError(f"Batch size {len(batch)} and indices size {len(indices)} mismatch.")

        rew = buffer.rew
        bsz = len(indices)
        indices = [indices]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indices',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return cast(BatchWithReturnsProtocol, batch)

    @staticmethod
    def _compile() -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        _gae_return(f64, f64, f64, b, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)


# TODO: rename? See docstring
@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    r"""Computes advantages with GAE.
    """
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = rew[now].reshape(bsz, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)
