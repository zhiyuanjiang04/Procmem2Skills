#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alfworld.agents.environment import get_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a small real ALFWorld text trajectory.")
    parser.add_argument("--root", type=Path, required=True, help="Project root containing .cache/alfworld.")
    parser.add_argument("--output", type=Path, required=True, help="Raw JSON output path.")
    parser.add_argument("--split", default="train", choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--task-types", default="1", help="Comma-separated ALFWorld task type ids.")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-games", type=int, default=1)
    return parser.parse_args()


def load_task_metadata(gamefile: Path) -> dict:
    traj_data_path = gamefile.with_name("traj_data.json")
    if not traj_data_path.exists():
        return {}
    with traj_data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    annotations = ((payload.get("turk_annotations") or {}).get("anns") or [])
    task_desc = None
    if annotations:
        task_desc = annotations[0].get("task_desc")
    scene = payload.get("scene") or {}
    return {
        "task_type": payload.get("task_type"),
        "instruction": task_desc,
        "scene": scene.get("floor_plan"),
    }


def build_config(root: Path, split: str, task_types: list[int], num_games: int, max_steps: int) -> dict:
    data_root = root / ".cache/alfworld/json_2.1.1"
    return {
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "task_types": task_types,
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "dataset": {
            "data_path": str(data_root / "train"),
            "eval_id_data_path": str(data_root / "valid_seen"),
            "eval_ood_data_path": str(data_root / "valid_unseen"),
            "num_train_games": num_games,
            "num_eval_games": num_games,
        },
        "logic": {
            "domain": str(root / ".cache/alfworld/logic/alfred.pddl"),
            "grammar": str(root / ".cache/alfworld/logic/alfred.twl2"),
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


def main() -> None:
    args = parse_args()
    task_types = [int(item.strip()) for item in args.task_types.split(",") if item.strip()]
    config = build_config(args.root, args.split, task_types, args.num_games, args.max_steps)

    env_cls = get_environment("AlfredTWEnv")
    wrapper = env_cls(config, train_eval=args.split)
    env = wrapper.init_env(batch_size=1)

    observations, infos = env.reset()
    current_observation = str(observations[0]).strip()
    gamefile = Path(infos["extra.gamefile"][0])
    task_metadata = load_task_metadata(gamefile)
    instruction = task_metadata.get("instruction")
    if not instruction and "Your task is to:" in current_observation:
        instruction = current_observation.split("Your task is to:", 1)[1].strip().splitlines()[0]

    steps = []
    success = False
    for _ in range(args.max_steps):
        expert_plan = infos.get("extra.expert_plan") or [[]]
        admissible = infos.get("admissible_commands") or [[]]
        plan_candidates = expert_plan[0] if expert_plan else []
        command = plan_candidates[0] if plan_candidates else (admissible[0][0] if admissible and admissible[0] else "look")

        next_observations, rewards, dones, infos = env.step([command])
        reward = float(rewards[0]) if rewards else 0.0
        done = bool(dones[0]) if dones else False
        next_observation = str(next_observations[0]).strip()

        steps.append(
            {
                "summary": f"Expert text-world action: {command}",
                "observation": current_observation,
                "action": command,
                "feedback": next_observation,
                "reward": reward,
                "success_signal": "task completed" if done else None,
            }
        )
        current_observation = next_observation
        success = done
        if done:
            break

    record = {
        "episode_id": gamefile.parent.name,
        "task_id": gamefile.parent.parent.name,
        "instruction": instruction or "ALFWorld task",
        "task_type": task_metadata.get("task_type"),
        "scene": task_metadata.get("scene"),
        "split": args.split,
        "success": success,
        "score": 1.0 if success else 0.0,
        "steps": steps,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump([record], handle, indent=2)

    print(f"Wrote 1 ALFWorld trajectory to {args.output}")
    print(f"Task: {record['task_id']} | success={record['success']} | steps={len(steps)}")


if __name__ == "__main__":
    main()
