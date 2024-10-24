import logging
import pickle

from env import BaseVectorEnv, VectorEnvNormObs
from utils.highlevel.env import (
    ContinuousEnvironments,
    EnvFactoryRegistered,
    EnvMode,
    EnvPoolFactory,
    VectorEnvType,
)
from utils.highlevel.persistence import Persistence, PersistEvent, RestoreEvent
from utils.highlevel.world import World

envpool_is_available = True
try:
    import envpool
except ImportError:
    envpool_is_available = False
    envpool = None

log = logging.getLogger(__name__)


def make_mujoco_env(task: str, seed: int, num_train_envs: int, num_test_envs: int, obs_norm: bool):
    envs = MujocoEnvFactory(task, seed, obs_norm=obs_norm).create_envs(
        num_train_envs,
        num_test_envs,
    )
    return envs.env, envs.train_envs, envs.test_envs


class MujocoEnvObsRmsPersistence(Persistence):
    FILENAME = "env_obs_rms.pkl"

    def persist(self, event: PersistEvent, world: World) -> None:
        if event != PersistEvent.PERSIST_POLICY:
            return
        obs_rms = world.envs.train_envs.get_obs_rms()
        path = world.persist_path(self.FILENAME)
        log.info(f"Saving environment obs_rms value to {path}")
        with open(path, "wb") as f:
            pickle.dump(obs_rms, f)

    def restore(self, event: RestoreEvent, world: World):
        if event != RestoreEvent.RESTORE_POLICY:
            return
        path = world.restore_path(self.FILENAME)
        log.info(f"Restoring environment obs_rms value from {path}")
        with open(path, "rb") as f:
            obs_rms = pickle.load(f)
        world.envs.train_envs.set_obs_rms(obs_rms)
        world.envs.test_envs.set_obs_rms(obs_rms)
        if world.envs.watch_env is not None:
            world.envs.watch_env.set_obs_rms(obs_rms)


class MujocoEnvFactory(EnvFactoryRegistered):
    def __init__(
        self,
        task: str,
        seed: int,
        obs_norm: bool = True,
        venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM,
    ) -> None:
        super().__init__(
            task=task,
            seed=seed,
            venv_type=venv_type,
            envpool_factory=EnvPoolFactory() if envpool_is_available else None,
        )
        self.obs_norm = obs_norm

    def create_venv(self, num_envs: int, mode: EnvMode) -> BaseVectorEnv:
        """Create vectorized environments.

        :param num_envs: the number of environments
        :param mode: the mode for which to create
        :return: the vectorized environments
        """
        env = super().create_venv(num_envs, mode)
        # obs norm wrapper
        if self.obs_norm:
            env = VectorEnvNormObs(env, update_obs_rms=mode == EnvMode.TRAIN)
        return env

    def create_envs(
        self,
        num_training_envs: int,
        num_test_envs: int,
        create_watch_env: bool = False,
    ) -> ContinuousEnvironments:
        envs = super().create_envs(num_training_envs, num_test_envs, create_watch_env)
        assert isinstance(envs, ContinuousEnvironments)

        if self.obs_norm:
            envs.test_envs.set_obs_rms(envs.train_envs.get_obs_rms())
            if envs.watch_env is not None:
                envs.watch_env.set_obs_rms(envs.train_envs.get_obs_rms())
            envs.set_persistence(MujocoEnvObsRmsPersistence())
        return envs