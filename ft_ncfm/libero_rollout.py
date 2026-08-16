"""Simulator-backed downstream evaluation on fixed official LIBERO states."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .downstream import ActionRegressor
from .libero_data import process_libero_rgb, tokenize_instruction
from .libero_models import LiberoVLAPolicy


def _observation_tensors(
    observation: dict[str, np.ndarray], instruction_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.cat(
        [
            process_libero_rgb(observation["agentview_image"]),
            process_libero_rgb(observation["robot0_eye_in_hand_image"]),
        ]
    ).unsqueeze(0)
    proprio = torch.cat(
        [
            torch.as_tensor(observation["robot0_gripper_qpos"], dtype=torch.float32),
            torch.as_tensor(observation["robot0_joint_pos"], dtype=torch.float32),
        ]
    ).unsqueeze(0)
    return image, instruction_tokens.unsqueeze(0), proprio


def _load_models(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[LiberoVLAPolicy, dict[str, ActionRegressor], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = LiberoVLAPolicy(**payload["policy_arguments"])
    policy.load_state_dict(payload["guide_state_dict"])
    policy.to(device).eval()
    heads: dict[str, ActionRegressor] = {}
    for name, state in payload["action_heads"].items():
        head = ActionRegressor(
            payload["representation_dim"],
            payload["action_dim"],
            payload["downstream_hidden_dim"],
        )
        head.load_state_dict(state)
        heads[name] = head.to(device).eval()
    return policy, heads, payload


def evaluate_libero_checkpoint(
    checkpoint_path: str | Path,
    *,
    libero_root: str | Path,
    suite: str,
    task_name: str,
    instruction: str,
    variants: list[str] | None = None,
    episodes: int = 5,
    max_steps: int = 600,
    image_size: int = 128,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate policies using the benchmark's fixed initial simulator states."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise RuntimeError(
            "Simulator evaluation requires the pinned official LIBERO environment"
        ) from error

    run_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    guide, heads, checkpoint = _load_models(checkpoint_path, run_device)
    requested = variants or list(heads)
    missing = sorted(set(requested).difference(heads))
    if missing:
        raise ValueError(f"Checkpoint does not contain variants: {missing}")

    root = Path(libero_root).resolve()
    bddl_file = (
        root / "libero" / "libero" / "bddl_files" / suite / f"{task_name}.bddl"
    )
    init_file = (
        root / "libero" / "libero" / "init_files" / suite / f"{task_name}.pruned_init"
    )
    if not bddl_file.is_file() or not init_file.is_file():
        raise FileNotFoundError(
            f"Pinned task files are missing: bddl={bddl_file}, init={init_file}"
        )
    init_states = torch.load(init_file, map_location="cpu", weights_only=False)
    if episodes > len(init_states):
        raise ValueError("Requested more episodes than official fixed initial states")

    instruction_tokens = tokenize_instruction(
        instruction, max_length=int(checkpoint["instruction_length"])
    )
    result: dict[str, Any] = {
        "protocol": "official fixed initial states; five zero-action settling steps",
        "suite": suite,
        "task_name": task_name,
        "episodes": episodes,
        "max_steps": max_steps,
        "variants": {},
    }
    for variant in requested:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=image_size,
            camera_widths=image_size,
        )
        env.seed(seed)
        outcomes: list[dict[str, int | bool]] = []
        try:
            for episode in range(episodes):
                env.reset()
                observation = env.set_init_state(init_states[episode])
                for _ in range(5):
                    observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
                success = False
                steps_taken = 0
                with torch.no_grad():
                    for step in range(1, max_steps + 1):
                        steps_taken = step
                        image, tokens, proprio = _observation_tensors(
                            observation, instruction_tokens
                        )
                        representation = guide.encode(
                            image.to(run_device),
                            tokens.to(run_device),
                            proprio.to(run_device),
                        )
                        action = (
                            heads[variant](representation)
                            .clamp(-1.0, 1.0)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )
                        observation, _, done, _ = env.step(action)
                        success = bool(done or env.check_success())
                        if success:
                            break
                outcomes.append(
                    {
                        "init_state_index": episode,
                        "success": success,
                        "steps": steps_taken,
                    }
                )
        finally:
            env.close()
            gc.collect()
        successes = sum(int(outcome["success"]) for outcome in outcomes)
        result["variants"][variant] = {
            "successes": successes,
            "success_rate": successes / episodes,
            "outcomes": outcomes,
        }
    return result
