from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    metadata: dict


class DatasetAdapter:
    def __init__(self, dataset_id: str, download: bool = True) -> None:
        import minari

        self.dataset_id = dataset_id
        self.dataset = minari.load_dataset(dataset_id, download=download)

    @property
    def total_episodes(self) -> int:
        return self.dataset.total_episodes

    @property
    def total_steps(self) -> int:
        return self.dataset.total_steps

    def get_episodes(
        self,
        max_episodes: int | None = None,
        obs_key: str = "observation",
    ) -> list[Episode]:
        episodes = []
        for i, ep in enumerate(self.dataset.iterate_episodes()):
            if max_episodes is not None and i >= max_episodes:
                break

            if isinstance(ep.observations, dict):
                obs = np.asarray(ep.observations[obs_key], dtype=np.float32)
                metadata = {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in ep.observations.items()
                    if k != obs_key
                }
            else:
                obs = np.asarray(ep.observations, dtype=np.float32)
                metadata = {}

            episodes.append(Episode(
                observations=obs,
                actions=np.asarray(ep.actions, dtype=np.float32),
                rewards=np.asarray(ep.rewards, dtype=np.float32),
                terminations=np.asarray(ep.terminations),
                truncations=np.asarray(ep.truncations),
                metadata=metadata,
            ))

        return episodes

    def get_all_observations(
        self,
        max_episodes: int | None = None,
        obs_key: str = "observation",
    ) -> np.ndarray:
        episodes = self.get_episodes(max_episodes=max_episodes, obs_key=obs_key)
        return np.concatenate([ep.observations for ep in episodes], axis=0)

    def get_obs_dim(self, obs_key: str = "observation") -> int:
        ep = next(self.dataset.iterate_episodes())
        if isinstance(ep.observations, dict):
            return ep.observations[obs_key].shape[-1]
        return ep.observations.shape[-1]

    def get_action_dim(self) -> int:
        ep = next(self.dataset.iterate_episodes())
        return ep.actions.shape[-1]

    def observation_statistics(
        self,
        max_episodes: int | None = 500,
        obs_key: str = "observation",
    ) -> dict:
        all_obs = self.get_all_observations(
            max_episodes=max_episodes, obs_key=obs_key,
        )

        return {
            "mean": np.mean(all_obs, axis=0),
            "std": np.std(all_obs, axis=0),
            "min": np.min(all_obs, axis=0),
            "max": np.max(all_obs, axis=0),
            "percentiles": {
                p: np.percentile(all_obs, p, axis=0)
                for p in [10, 25, 30, 50, 70, 75, 90]
            },
            "num_samples": len(all_obs),
        }
