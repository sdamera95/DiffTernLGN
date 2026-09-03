from __future__ import annotations

import numpy as np

from pst_dtlgn.applications.stl.predicates import Predicate, PredicateModule
from pst_dtlgn.applications.stl.stl_labeler import (
    STLSpec,
    OnlineSTLSpec,
    stl_always,
    stl_eventually,
    stl_and,
    stl_or,
    stl_not,
    stl_implies,
    stl_always_online,
    stl_eventually_online,
)


# PointMaze / Maze2D
# State: 4D (x, y, vx, vy)

def pointmaze_predicates(
    goal_pos: np.ndarray,
    r_goal: float = 0.5,
    r_safe: float = 0.3,
    v_min: float = 0.1,
    v_max: float = 3.0,
    theta_at_goal: tuple[float, float] = (-0.5, 0.0),
    theta_safe: tuple[float, float] = (0.0, 0.3),
    theta_moving: tuple[float, float] = (-0.05, 0.05),
    theta_bounded_speed: tuple[float, float] = (-0.5, 0.5),
) -> list[Predicate]:
    goal = np.asarray(goal_pos, dtype=np.float32)

    def at_goal_feature(obs):
        pos = obs[:2]
        return r_goal - np.sqrt(np.sum((pos - goal) ** 2))

    def safe_feature(obs):
        x, y = float(obs[0]), float(obs[1])
        wall_bound = 1.3
        d_wall = min(
            wall_bound - abs(x),
            wall_bound - abs(y),
        )
        return d_wall - r_safe

    def moving_feature(obs):
        vel = obs[2:4]
        return np.sqrt(np.sum(vel ** 2)) - v_min

    def bounded_speed_feature(obs):
        vel = obs[2:4]
        return v_max - np.sqrt(np.sum(vel ** 2))

    return [
        Predicate(
            name="at_goal",
            feature_fn=at_goal_feature,
            theta_low=theta_at_goal[0],
            theta_high=theta_at_goal[1],
            description=f"Within radius {r_goal} of goal {goal_pos}",
        ),
        Predicate(
            name="safe",
            feature_fn=safe_feature,
            theta_low=theta_safe[0],
            theta_high=theta_safe[1],
            description=f"Distance from walls > {r_safe}",
        ),
        Predicate(
            name="moving",
            feature_fn=moving_feature,
            theta_low=theta_moving[0],
            theta_high=theta_moving[1],
            description=f"Speed above {v_min}",
        ),
        Predicate(
            name="bounded_speed",
            feature_fn=bounded_speed_feature,
            theta_low=theta_bounded_speed[0],
            theta_high=theta_bounded_speed[1],
            description=f"Speed below {v_max}",
        ),
    ]


def pointmaze_stl_specs(horizon: int | None = None) -> list[STLSpec]:
    def safety_eval(preds):
        return stl_always(preds[:, 1], horizon)

    def reach_eval(preds):
        return stl_eventually(preds[:, 0], horizon)

    def reach_avoid_eval(preds):
        return stl_and(
            stl_always(preds[:, 1], horizon),
            stl_eventually(preds[:, 0], horizon),
        )

    def speed_limit_eval(preds):
        return stl_always(preds[:, 3], horizon)

    return [
        STLSpec("safety", "□ safe", safety_eval, "safety"),
        STLSpec("reach", "◇ at_goal", reach_eval, "reachability"),
        STLSpec("reach_avoid", "□ safe ∧ ◇ at_goal", reach_avoid_eval, "combined"),
        STLSpec("speed_limit", "□ bounded_speed", speed_limit_eval, "safety"),
    ]


def pointmaze_online_specs(horizon: int | None = None) -> list[OnlineSTLSpec]:
    def safety_online(preds):
        return stl_always_online(preds[:, 1], horizon)

    def reach_online(preds):
        return stl_eventually_online(preds[:, 0], horizon)

    def reach_avoid_online(preds):
        return stl_and(
            stl_always_online(preds[:, 1], horizon),
            stl_eventually_online(preds[:, 0], horizon),
        )

    def speed_limit_online(preds):
        return stl_always_online(preds[:, 3], horizon)

    return [
        OnlineSTLSpec("safety_online", "□ safe", safety_online, "safety"),
        OnlineSTLSpec("reach_online", "◇ at_goal", reach_online, "reachability"),
        OnlineSTLSpec("reach_avoid_online", "□ safe ∧ ◇ at_goal",
                      reach_avoid_online, "combined"),
        OnlineSTLSpec("speed_limit_online", "□ bounded_speed",
                      speed_limit_online, "safety"),
    ]


# AntMaze
# State: 29D (qpos + qvel)

def antmaze_predicates(
    goal_pos: np.ndarray,
    r_goal: float = 1.0,
    z_min: float = 0.3,
    omega_max: float = 2.0,
    v_min: float = 0.1,
    v_max: float = 5.0,
    theta_at_goal: tuple[float, float] = (-1.0, 0.0),
    theta_upright: tuple[float, float] = (-0.1, 0.1),
    theta_stable: tuple[float, float] = (-0.5, 0.5),
    theta_moving: tuple[float, float] = (-0.05, 0.05),
    theta_bounded_speed: tuple[float, float] = (-0.5, 0.5),
    theta_safe: tuple[float, float] = (0.0, 0.3),
) -> list[Predicate]:
    goal = np.asarray(goal_pos, dtype=np.float32)

    def at_goal_feature(obs):
        pos = obs[:2]
        return r_goal - np.sqrt(np.sum((pos - goal) ** 2))

    def upright_feature(obs):
        z = float(obs[2])
        return z - z_min

    def stable_feature(obs):
        omega = obs[17:20]
        return omega_max - np.sqrt(np.sum(omega ** 2))

    def moving_feature(obs):
        vel = obs[15:17]
        return np.sqrt(np.sum(vel ** 2)) - v_min

    def bounded_speed_feature(obs):
        vel = obs[15:17]
        return v_max - np.sqrt(np.sum(vel ** 2))

    def safe_feature(obs):
        x, y = float(obs[0]), float(obs[1])
        wall_bound = 3.0
        d_wall = min(wall_bound - abs(x), wall_bound - abs(y))
        return d_wall - 0.5

    return [
        Predicate("at_goal", at_goal_feature, *theta_at_goal,
                  f"Within {r_goal} of goal"),
        Predicate("safe", safe_feature, *theta_safe,
                  "Distance from walls"),
        Predicate("moving", moving_feature, *theta_moving,
                  f"Speed above {v_min}"),
        Predicate("bounded_speed", bounded_speed_feature, *theta_bounded_speed,
                  f"Speed below {v_max}"),
        Predicate("upright", upright_feature, *theta_upright,
                  f"Torso height above {z_min}"),
        Predicate("stable", stable_feature, *theta_stable,
                  f"Angular velocity below {omega_max}"),
    ]


def antmaze_stl_specs(horizon: int | None = None) -> list[STLSpec]:
    def safety_eval(preds):
        return stl_always(preds[:, 1], horizon)

    def reach_eval(preds):
        return stl_eventually(preds[:, 0], horizon)

    def reach_avoid_eval(preds):
        return stl_and(
            stl_always(preds[:, 1], horizon),
            stl_eventually(preds[:, 0], horizon),
        )

    def speed_limit_eval(preds):
        return stl_always(preds[:, 3], horizon)

    def liveness_eval(preds):
        return stl_always(stl_eventually(preds[:, 2], horizon), horizon)

    def response_eval(preds):
        return stl_always(
            stl_implies(stl_not(preds[:, 1]), stl_eventually(preds[:, 1], horizon)),
            horizon,
        )

    return [
        STLSpec("safety", "□ safe", safety_eval, "safety"),
        STLSpec("reach", "◇ at_goal", reach_eval, "reachability"),
        STLSpec("reach_avoid", "□ safe ∧ ◇ at_goal", reach_avoid_eval, "combined"),
        STLSpec("speed_limit", "□ bounded_speed", speed_limit_eval, "safety"),
        STLSpec("liveness", "□ ◇ moving", liveness_eval, "liveness"),
        STLSpec("response", "□(¬safe → ◇ safe)", response_eval, "response"),
    ]


# Kitchen / Franka
# State: 60D (robot joints + object states)

def kitchen_predicates(
    theta_microwave: tuple[float, float] = (-0.1, 0.1),
    theta_kettle: tuple[float, float] = (-0.1, 0.1),
    theta_light: tuple[float, float] = (-0.1, 0.1),
    theta_cabinet: tuple[float, float] = (-0.1, 0.1),
    theta_gripper: tuple[float, float] = (-0.1, 0.1),
    theta_bounded_speed: tuple[float, float] = (-0.5, 0.5),
) -> list[Predicate]:
    MICROWAVE_IDX = 22
    KETTLE_POS_IDX = slice(23, 26)
    BURNER_POS = np.array([-0.269, 0.35, 1.626], dtype=np.float32)
    LIGHT_IDX = 27
    CABINET_IDX = 21

    def microwave_feature(obs):
        return float(obs[MICROWAVE_IDX]) - 0.5

    def kettle_feature(obs):
        kettle_pos = obs[KETTLE_POS_IDX]
        return 0.3 - np.sqrt(np.sum((kettle_pos - BURNER_POS) ** 2))

    def light_feature(obs):
        return float(obs[LIGHT_IDX]) - 0.5

    def cabinet_feature(obs):
        return float(obs[CABINET_IDX]) - 0.1

    def gripper_feature(obs):
        grip = (float(obs[7]) + float(obs[8])) / 2.0
        return grip - 0.01

    def bounded_speed_feature(obs):
        ee_vel = obs[9:12]
        return 2.0 - np.sqrt(np.sum(ee_vel ** 2))

    return [
        Predicate("microwave_open", microwave_feature, *theta_microwave,
                  "Microwave door open"),
        Predicate("kettle_on_burner", kettle_feature, *theta_kettle,
                  "Kettle on burner"),
        Predicate("light_on", light_feature, *theta_light,
                  "Light switch on"),
        Predicate("cabinet_open", cabinet_feature, *theta_cabinet,
                  "Cabinet door open"),
        Predicate("gripper_grasping", gripper_feature, *theta_gripper,
                  "Gripper is closed/grasping"),
        Predicate("bounded_speed", bounded_speed_feature, *theta_bounded_speed,
                  "End-effector speed below limit"),
    ]


def kitchen_stl_specs(horizon: int | None = None) -> list[STLSpec]:
    def sequential_task_eval(preds):
        return stl_and(
            stl_eventually(preds[:, 0], horizon),
            stl_always(
                stl_implies(preds[:, 0], stl_eventually(preds[:, 1], horizon)),
                horizon,
            ),
        )

    def safe_grasp_eval(preds):
        return stl_always(
            stl_implies(preds[:, 4], preds[:, 5]),
            horizon,
        )

    def full_ordering_eval(preds):
        step1 = stl_eventually(preds[:, 0], horizon)
        step2 = stl_always(
            stl_implies(preds[:, 0], stl_eventually(preds[:, 1], horizon)),
            horizon,
        )
        step3 = stl_always(
            stl_implies(preds[:, 1], stl_eventually(preds[:, 2], horizon)),
            horizon,
        )
        return stl_and(step1, stl_and(step2, step3))

    return [
        STLSpec("sequential_task", "◇ micro ∧ □(micro → ◇ kettle)",
                sequential_task_eval, "response"),
        STLSpec("safe_grasp", "□(grasp → bounded_speed)",
                safe_grasp_eval, "safety"),
        STLSpec("full_ordering", "◇ micro ∧ □(micro→◇kettle) ∧ □(kettle→◇light)",
                full_ordering_eval, "response"),
    ]


# Environment registry

ENVIRONMENT_CONFIGS = {
    "pointmaze_umaze": {
        "minari_id": "D4RL/pointmaze/umaze-v2",
        "obs_key": "observation",
        "obs_dim": 4,
        "action_dim": 2,
        "predicates_fn": pointmaze_predicates,
        "specs_fn": pointmaze_stl_specs,
        "online_specs_fn": pointmaze_online_specs,
        "description": "PointMaze U-maze (Phase 1 validation)",
    },
    "pointmaze_medium": {
        "minari_id": "D4RL/pointmaze/medium-v2",
        "obs_key": "observation",
        "obs_dim": 4,
        "action_dim": 2,
        "predicates_fn": pointmaze_predicates,
        "specs_fn": pointmaze_stl_specs,
        "online_specs_fn": pointmaze_online_specs,
        "description": "PointMaze Medium (Phase 1 validation)",
    },
    "antmaze_medium": {
        "minari_id": "D4RL/antmaze/medium-diverse-v1",
        "obs_key": "observation",
        "obs_dim": 29,
        "action_dim": 8,
        "predicates_fn": antmaze_predicates,
        "specs_fn": antmaze_stl_specs,
        "description": "AntMaze Medium Diverse (Phase 2 scaling)",
    },
    "kitchen_mixed": {
        "minari_id": "D4RL/kitchen/mixed-v2",
        "obs_key": "observation",
        "obs_dim": 60,
        "action_dim": 9,
        "predicates_fn": kitchen_predicates,
        "specs_fn": kitchen_stl_specs,
        "description": "Kitchen Mixed (Phase 3 demonstration)",
    },
}
