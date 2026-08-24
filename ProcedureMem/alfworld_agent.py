"""Reliable single- and multi-task execution helpers for ALFWorld."""

from __future__ import annotations

import copy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


ACTION_PATTERNS = (
    re.compile(r"^look$"),
    re.compile(r"^inventory$"),
    re.compile(r"^examine .+$"),
    re.compile(r"^go to .+$"),
    re.compile(r"^take .+ from .+$"),
    re.compile(r"^move .+ to .+$"),
    re.compile(r"^open .+$"),
    re.compile(r"^close .+$"),
    re.compile(r"^use .+$"),
    re.compile(r"^clean .+ with .+$"),
    re.compile(r"^heat .+ with .+$"),
    re.compile(r"^cool .+ with .+$"),
)
ACTION_LABEL = re.compile(r"\baction\s*:\s*([^\r\n]+)", re.IGNORECASE)
INACTIVE_ACTION = "look"


def resolve_litellm_model(model: str, api_base: str | None) -> str:
    """Route custom OpenAI-compatible endpoints through LiteLLM's provider."""
    if api_base and not model.startswith("openai/"):
        return f"openai/{model}"
    return model


def _clean_action(candidate: str) -> str:
    action = candidate.strip().strip("`*\"").strip().rstrip(".")
    action = re.sub(r"\s+", " ", action).lower()
    put_match = re.fullmatch(r"put (.+) in/on (.+)", action)
    if put_match:
        return f"move {put_match.group(1)} to {put_match.group(2)}"
    toggle_match = re.fullmatch(r"toggle (.+)", action)
    if toggle_match:
        return f"use {toggle_match.group(1)}"
    return action


def is_valid_action(action: str) -> bool:
    """Return whether an action follows the ALFWorld command templates."""
    return any(pattern.fullmatch(action) for pattern in ACTION_PATTERNS)


def parse_action(response: str) -> str:
    """Extract an action while leaving validity feedback to ALFWorld."""
    if not isinstance(response, str) or not response.strip():
        return ""

    labelled = [_clean_action(value) for value in ACTION_LABEL.findall(response)]
    if labelled:
        return labelled[-1]

    nonempty_lines = [
        _clean_action(line)
        for line in response.splitlines()
        if line.strip() and not line.lstrip().startswith("```")
    ]
    if len(nonempty_lines) == 1 and is_valid_action(nonempty_lines[0]):
        return nonempty_lines[0]
    return ""


def process_observation(observation: str) -> str:
    if observation.startswith("You arrive at loc "):
        separator = observation.find(". ")
        if separator >= 0:
            return observation[separator + 2 :]
    return observation


def get_example(name: str, examples: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }
    for prefix, task_type in prefixes.items():
        if name.startswith(prefix):
            for example in examples:
                if example["task"] == task_type:
                    return copy.deepcopy(example["example"])
    raise ValueError(f"No few-shot example found for ALFWorld task {name!r}")


def build_messages(
    observation: str,
    name: str,
    *,
    system_prompt: str,
    few_shot: bool,
    examples: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if few_shot:
        example = get_example(name, examples)
        example[0]["content"] = (
            "Here is an example of how to solve the task:\nExample:\n"
            + example[0]["content"]
        )
        messages.extend(example)
        messages.append({"role": "user", "content": "Now it's your turn.\n" + observation})
    else:
        messages.append({"role": "user", "content": observation})
    return messages


@dataclass
class TaskState:
    name: str
    messages: list[dict[str, str]]
    trajectory: list[dict[str, str]] = field(default_factory=list)
    active: bool = True
    reward: bool = False
    steps: int = 0
    termination_reason: str | None = None
    error: str | None = None
    actions: list[str] = field(default_factory=list)

    def finish(self, reason: str, *, error: str | None = None) -> None:
        self.active = False
        self.termination_reason = reason
        self.error = error

    def as_result(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "reward": self.reward,
            "name": self.name,
            "steps": self.steps,
            "termination_reason": self.termination_reason,
            "error": self.error,
            "actions": self.actions,
            "trajectory": self.trajectory,
        }


def run_alfworld_batch(
    *,
    env: Any,
    observations: Sequence[str],
    trajectory_observations: Sequence[str] | None = None,
    names: Sequence[str],
    llm_fn: Callable[[list[dict[str, str]]], str],
    system_prompt: str,
    few_shot: bool = True,
    max_steps: int = 30,
    examples: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Run a batch while isolating completion and failure state per task."""
    if len(observations) != len(names):
        raise ValueError("observations and names must have the same length")
    if trajectory_observations is None:
        trajectory_observations = observations
    if len(trajectory_observations) != len(observations):
        raise ValueError(
            "trajectory_observations and observations must have the same length"
        )
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not observations:
        return []

    states = [
        TaskState(
            name=name,
            messages=build_messages(
                observation,
                name,
                system_prompt=system_prompt,
                few_shot=few_shot,
                examples=examples,
            ),
            trajectory=[{"from": "human", "value": trajectory_observation}],
        )
        for observation, trajectory_observation, name in zip(
            observations, trajectory_observations, names
        )
    ]

    for _ in range(max_steps):
        active_indices = [index for index, state in enumerate(states) if state.active]
        if not active_indices:
            break

        responses: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=len(active_indices)) as executor:
            futures = {
                executor.submit(llm_fn, states[index].messages): index
                for index in active_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    response = future.result()
                    if not isinstance(response, str):
                        raise TypeError(f"LLM returned {type(response).__name__}, expected str")
                    responses[index] = response
                    states[index].messages.append(
                        {"role": "assistant", "content": response}
                    )
                    states[index].trajectory.append(
                        {"from": "gpt", "value": response}
                    )
                except Exception as exc:
                    states[index].finish("llm_error", error=str(exc))

        actions: dict[int, str] = {}
        for index, response in responses.items():
            if not states[index].active:
                continue
            actions[index] = parse_action(response)

        if not actions:
            continue

        action_list = [INACTIVE_ACTION] * len(states)
        for index, action in actions.items():
            action_list[index] = action

        try:
            observations_out, _, done, info = env.step(action_list)
        except Exception as exc:
            for index in actions:
                states[index].finish("environment_error", error=str(exc))
            continue

        won = info.get("won", [False] * len(states))
        for index, action in actions.items():
            state = states[index]
            state.steps += 1
            state.actions.append(action)
            state.reward = bool(won[index])
            observation = process_observation(str(observations_out[index]))
            state.messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )
            state.trajectory.append(
                {"from": "human", "value": f"Observation: {observation}"}
            )
            if state.reward:
                state.finish("success")
            elif bool(done[index]):
                state.finish("environment_done")

    for state in states:
        if state.active:
            state.finish("max_steps")
    return [state.as_result() for state in states]
