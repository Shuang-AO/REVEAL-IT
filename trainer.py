import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict
from plot_policy import visualization_policy

import numpy as np
import tqdm

from data import (
    AsyncCollector,
    Collector,
    CollectStats,
    EpochStats,
    InfoStats,
    ReplayBuffer,
    SequenceSummaryStats,
)
from data.collector import CollectStatsBase
from base import BasePolicy
from base import TrainingStats
from utils.trainer_utils import gather_info, test_episode
from utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    MovAvg,
    deprecation,
    tqdm_config,
)
from utils.logging import set_numerical_fields_to_precision

log = logging.getLogger(__name__)


class BaseTrainer(ABC):

    __doc__: str

    @staticmethod
    def gen_doc(learning_type: str) -> str:
        """Document string for subclass trainer."""
        step_means = f'The "step" in {learning_type} trainer means '
        if learning_type != "offline":
            step_means += "an environment step (a.k.a. transition)."
        else:  # offline
            step_means += "a gradient step."

        trainer_name = learning_type.capitalize() + "Trainer"

        return f"""An iterator class for {learning_type} trainer procedure.

        Returns an iterator that yields a 3-tuple (epoch, stats, info) of
        train results on every epoch.

        {step_means}

        Example usage:
            trainer = {trainer_name}(...)
            for epoch, epoch_stat, info in trainer:
                print("Epoch:", epoch)
                print(epoch_stat)
                print(info)
                do_something_with_policy()
                query_something_about_policy()
                make_a_plot_with(epoch_stat)
                display(info)
        """

    def __init__(
        self,
        policy: BasePolicy,
        max_epoch: int,
        batch_size: int | None,
        train_collector: Collector | None = None,
        test_collector: Collector | None = None,
        buffer: ReplayBuffer | None = None,
        step_per_epoch: int | None = None,
        repeat_per_collect: int | None = None,
        episode_per_test: int | None = None,
        update_per_step: float = 1.0,
        step_per_collect: int | None = None,
        episode_per_collect: int | None = None,
        train_fn: Callable[[int, int], None] | None = None,
        test_fn: Callable[[int, int | None], None] | None = None,
        stop_fn: Callable[[float], bool] | None = None,
        save_best_fn: Callable[[BasePolicy], None] | None = None,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
        resume_from_log: bool = False,
        reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        save_fn: Callable[[BasePolicy], None] | None = None,
    ):
        if save_fn:
            deprecation(
                "save_fn in trainer is marked as deprecated and will be "
                "removed in the future. Please use save_best_fn instead.",
            )
            assert save_best_fn is None
            save_best_fn = save_fn

        self.policy = policy

        if buffer is not None:
            buffer = policy.process_buffer(buffer)
        self.buffer = buffer

        self.train_collector = train_collector
        self.test_collector = test_collector

        self.logger = logger
        self.start_time = time.time()
        self.stat: defaultdict[str, MovAvg] = defaultdict(MovAvg)
        self.best_reward = 0.0
        self.best_reward_std = 0.0
        self.start_epoch = 0
        # This is only used for logging but creeps into the implementations
        # of the trainers. I believe it would be better to remove
        self._gradient_step = 0
        self.env_step = 0
        self.policy_update_time = 0.0
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch

        # either on of these two
        self.step_per_collect = step_per_collect
        self.episode_per_collect = episode_per_collect

        self.update_per_step = update_per_step
        self.repeat_per_collect = repeat_per_collect

        self.episode_per_test = episode_per_test

        self.batch_size = batch_size

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.stop_fn = stop_fn
        self.save_best_fn = save_best_fn
        self.save_checkpoint_fn = save_checkpoint_fn

        self.reward_metric = reward_metric
        self.verbose = verbose
        self.show_progress = show_progress
        self.test_in_train = test_in_train
        self.resume_from_log = resume_from_log

        self.is_run = False
        self.last_rew, self.last_len = 0.0, 0.0

        self.epoch = self.start_epoch
        self.best_epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        if self.resume_from_log:
            (
                self.start_epoch,
                self.env_step,
                self._gradient_step,
            ) = self.logger.restore_data()

        self.last_rew, self.last_len = 0.0, 0.0
        self.start_time = time.time()
        if self.train_collector is not None:
            self.train_collector.reset_stat()

            if self.train_collector.policy != self.policy or self.test_collector is None:
                self.test_in_train = False

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            assert not isinstance(self.test_collector, AsyncCollector)  # Issue 700
            self.test_collector.reset_stat()
            test_result = test_episode(
                self.policy,
                self.test_collector,
                self.test_fn,
                self.start_epoch,
                self.episode_per_test,
                self.logger,
                self.env_step,
                self.reward_metric,
            )
            assert test_result.returns_stat is not None  # for mypy
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = (
                test_result.returns_stat.mean,
                test_result.returns_stat.std,
            )
        if self.save_best_fn:
            self.save_best_fn(self.policy)

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> EpochStats:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        progress = tqdm.tqdm if self.show_progress else DummyTqdm

        # perform n step_per_epoch
        with progress(total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config) as t:
            while t.n < t.total and not self.stop_fn_flag:
                train_stat: CollectStatsBase
                if self.train_collector is not None:
                    train_stat, self.stop_fn_flag = self.train_step()
                    pbar_data_dict = {
                        "env_step": str(self.env_step),
                        "rew": f"{self.last_rew:.2f}",
                        "len": str(int(self.last_len)),
                        "n/ep": str(train_stat.n_collected_episodes),
                        "n/st": str(train_stat.n_collected_steps),
                    }
                    t.update(train_stat.n_collected_steps)
                    if self.stop_fn_flag:
                        t.set_postfix(**pbar_data_dict)
                        break
                else:
                    pbar_data_dict = {}
                    assert self.buffer, "No train_collector or buffer specified"
                    train_stat = CollectStatsBase(
                        n_collected_episodes=len(self.buffer),
                    )
                    t.update()

                update_stat = self.policy_update_fn(train_stat)
                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["gradient_step"] = str(self._gradient_step)

                t.set_postfix(**pbar_data_dict)

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        test_stat = None
        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch,
                self.env_step,
                self._gradient_step,
                self.save_checkpoint_fn,
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()

        info_stat = gather_info(
            start_time=self.start_time,
            policy_update_time=self.policy_update_time,
            gradient_step=self._gradient_step,
            best_reward=self.best_reward,
            best_reward_std=self.best_reward_std,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
        )

        self.logger.log_info_data(asdict(info_stat), self.epoch)

        # in case trainer is used with run(), epoch_stat will not be returned
        epoch_stat: EpochStats = EpochStats(
            epoch=self.epoch,
            train_collect_stat=train_stat,
            test_collect_stat=test_stat,
            training_stat=update_stat,
            info_stat=info_stat,
        )

        return epoch_stat

    def test_step(self) -> tuple[CollectStats, bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_stat = test_episode(
            self.policy,
            self.test_collector,
            self.test_fn,
            self.epoch,
            self.episode_per_test,
            self.logger,
            self.env_step,
            self.reward_metric,
        )
        assert test_stat.returns_stat is not None  # for mypy
        rew, rew_std = test_stat.returns_stat.mean, test_stat.returns_stat.std
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        log_msg = (
            f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
            f" best_reward: {self.best_reward:.6f} ± "
            f"{self.best_reward_std:.6f} in #{self.best_epoch}"
        )
        log.info(log_msg)
        if self.verbose:
            print(log_msg, flush=True)

        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    def train_step(self) -> tuple[CollectStats, bool]:
        """Perform one training step.
        """
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        should_stop_training = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result = self.train_collector.collect(
            n_step=self.step_per_collect,
            n_episode=self.episode_per_collect,
        )

        self.env_step += result.n_collected_steps

        if result.n_collected_episodes > 0:
            assert result.returns_stat is not None  # for mypy
            assert result.lens_stat is not None  # for mypy
            self.last_rew = result.returns_stat.mean
            self.last_len = result.lens_stat.mean
            if self.reward_metric:  # TODO: move inside collector
                rew = self.reward_metric(result.returns)
                result.returns = rew
                result.returns_stat = SequenceSummaryStats.from_sequence(rew)

            self.logger.log_train_data(asdict(result), self.env_step)

        if (
            result.n_collected_episodes > 0
            and self.test_in_train
            and self.stop_fn
            and self.stop_fn(result.returns_stat.mean)  # type: ignore
        ):
            assert self.test_collector is not None
            test_result = test_episode(
                self.policy,
                self.test_collector,
                self.test_fn,
                self.epoch,
                self.episode_per_test,
                self.logger,
                self.env_step,
            )
            assert test_result.returns_stat is not None  # for mypy
            if self.stop_fn(test_result.returns_stat.mean):
                should_stop_training = True
                self.best_reward = test_result.returns_stat.mean
                self.best_reward_std = test_result.returns_stat.std
            else:
                self.policy.train()
        return result, should_stop_training

    # TODO: move moving average computation and logging into its own logger
    # TODO: maybe think about a command line logger instead of always printing data dict
    def _update_moving_avg_stats_and_log_update_data(self, update_stat: TrainingStats) -> None:
        """Log losses, update moving average stats, and also modify the smoothed_loss in update_stat."""
        cur_losses_dict = update_stat.get_loss_stats_dict()
        update_stat.smoothed_loss = self._update_moving_avg_stats_and_get_averaged_data(
            cur_losses_dict,
        )
        self.logger.log_update_data(asdict(update_stat), self._gradient_step)

    # TODO: seems convoluted, there should be a better way of dealing with the moving average stats
    def _update_moving_avg_stats_and_get_averaged_data(
        self,
        data: dict[str, float],
    ) -> dict[str, float]:
        """Add entries to the moving average object in the trainer and retrieve the averaged results.
        """
        smoothed_data = {}
        for key, loss_item in data.items():
            self.stat[key].add(loss_item)
            smoothed_data[key] = self.stat[key].get()
        return smoothed_data

    @abstractmethod
    def policy_update_fn(
        self,
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Policy update function for different trainer implementation.
        """

    def run(self) -> InfoStats:
        """Consume iterator.
        """
        try:
            self.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_info(
                start_time=self.start_time,
                policy_update_time=self.policy_update_time,
                gradient_step=self._gradient_step,
                best_reward=self.best_reward,
                best_reward_std=self.best_reward_std,
                train_collector=self.train_collector,
                test_collector=self.test_collector,
            )
        finally:
            self.is_run = False

        return info

    def _sample_and_update(self, buffer: ReplayBuffer) -> TrainingStats:
        """Sample a mini-batch, perform one gradient step, and update the _gradient_step counter."""
        self._gradient_step += 1
        # Note: since sample_size=batch_size, this will perform
        # exactly one gradient step. This is why we don't need to calculate the
        # number of gradient steps, like in the on-policy case.
        update_stat = self.policy.update(sample_size=self.batch_size, buffer=buffer)
        self._update_moving_avg_stats_and_log_update_data(update_stat)
        return update_stat


class OnpolicyTrainer(BaseTrainer):
    """On-policy trainer, passes the entire buffer to .update and resets it after.
    """

    # for mypy
    assert isinstance(BaseTrainer.__doc__, str)
    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(BaseTrainer.__doc__.split("\n")[1:])

    def policy_update_fn(
        self,
        result: CollectStatsBase | None = None,
    ) -> TrainingStats:
        """Perform one on-policy update by passing the entire buffer to the policy's update method."""
        assert self.train_collector is not None
        training_stat = self.policy.update(
            sample_size=0,
            buffer=self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )

        # just for logging, no functional role
        self.policy_update_time += training_stat.train_time

        self._gradient_step += 1
        if self.batch_size is None:
            self._gradient_step += 1
        elif self.batch_size > 0:
            self._gradient_step += int((len(self.train_collector.buffer) - 0.1) // self.batch_size)
        self.train_collector.reset_buffer(keep_statistics=True)
        visualization_policy(self.policy)

        # The step is the number of mini-batches used for the update, so essentially
        self._update_moving_avg_stats_and_log_update_data(training_stat)

        return training_stat