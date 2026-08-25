"""ALFWorld-specific prompts for building procedural memory.

These templates reconstruct the prompt omitted from the released MemP code. The
structure follows the draft accepted by a MemP author in GitHub issue #6; it is
not claimed to be the verbatim prompt used for the paper's original runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


PROMPT_DOMAIN = "alfworld"
PROMPT_VERSION = "issue-6-reconstruction-v1"
PROMPT_SOURCE = "https://github.com/zjunlp/MemP/issues/6#issuecomment-4194330680"
PROMPT_APPROVAL = "https://github.com/zjunlp/MemP/issues/6#issuecomment-4196611474"
SYSTEM_PROMPT = "You are a helpful assistant."


DIRECT_PROMPT_TEMPLATE = """
You are given an ALFWorld household task and a trajectory that attempts to solve it.
The trajectory contains multiple rounds of Thought, Action, and Observation.

Your task is to distill this trajectory into a reusable procedural workflow that can help solve similar ALFWorld tasks in the future.

Guidelines:
1. Focus on the critical steps that are necessary for task completion.
2. Summarize the procedure at a useful level of abstraction:
   - include important search heuristics such as likely locations or receptacles,
   - include required operation order such as pick -> clean -> place or pick -> heat -> place,
   - include important interaction constraints such as opening receptacles before taking objects,
   - omit redundant wandering, repeated failed attempts, and unhelpful low-value exploration.
3. The workflow should be domain-specific to household embodied tasks, not travel planning or general advice.
4. Write the workflow as a short, natural, coherent paragraph.
5. Do not copy the full trajectory verbatim.
6. Do not mention "Thought", "Observation", or step numbers.
7. If the trajectory fails, still summarize useful successful sub-procedures when possible, but avoid preserving clearly wrong behavior.

Examples of the workflow style:
- For pick-and-place: search likely receptacles for the target, take it, navigate to the destination, and place it there.
- For clean-and-place: find and take the target, go to a sink, clean it, then place it at the destination.
- For heat-and-place: find and take the target, heat it correctly with the microwave, then place it at the destination.
- For cool-and-place: find and take the target, cool it correctly with the fridge, then place it at the destination.
- For examine-in-light: find and take the target, navigate to a lamp, and use the lamp as required to examine it.
- For pick-two-objects: retrieve the two required objects one at a time and place both at the destination while tracking progress.

Task:
{query}

Trajectory:
{trajectory}

Output only the workflow paragraph without extra explanation:
""".strip()


EVENTS_PROMPT_TEMPLATE = """
You are given an ALFWorld trajectory consisting of multiple rounds of Thought, Action, and Observation.
Convert each round into an event in JSON format using this structure:

[
  {{
    "step": <step id>,
    "pre_state": "<state before the action>",
    "action": "<the action taken>",
    "entity": "agent",
    "new_state": "<state after the action>"
  }}
]

Rules:
1. Use ALFWorld household semantics.
2. Capture navigation, opening receptacles, taking objects, cleaning, heating, cooling, examining, and placing.
3. Represent failed or invalid actions clearly in the state transition.
4. Keep states concise but informative.

Task:
{query}

Trajectory:
{trajectory}

Output only the JSON:
""".strip()


WORKFLOW_FROM_EVENTS_PROMPT_TEMPLATE = """
You are given an ALFWorld task and a sequence of events extracted from a trajectory.
Each event contains step, pre_state, action, entity, and new_state.

Identify the critical events that are most important for solving the household task. Critical events usually include navigating to a useful receptacle, opening it when needed, taking the target object, performing required operations such as cleaning, heating, cooling, or examining, and placing the object at the destination. Omit redundant wandering and repeated failed attempts unless they reveal an essential interaction constraint.

Task:
{query}

Events:
{events}

Output only a JSON list of critical step ids, for example:
[1, 2, 4, 5]
""".strip()


@dataclass(frozen=True)
class PromptSpec:
    domain: str
    build_policy: str
    version: str
    sha256: str
    source: str
    approval: str

    def as_dict(self) -> dict[str, str]:
        return {
            "domain": self.domain,
            "build_policy": self.build_policy,
            "version": self.version,
            "sha256": self.sha256,
            "source": self.source,
            "approval": self.approval,
        }


def _prompt_material(build_policy: str) -> str:
    if build_policy == "direct":
        templates = (DIRECT_PROMPT_TEMPLATE,)
    elif build_policy == "round":
        templates = (EVENTS_PROMPT_TEMPLATE, WORKFLOW_FROM_EVENTS_PROMPT_TEMPLATE)
    else:
        raise ValueError(f"Unsupported ALFWorld build policy: {build_policy}")
    return "\n\n".join((SYSTEM_PROMPT, PROMPT_VERSION, *templates))


def get_prompt_spec(build_policy: str) -> PromptSpec:
    digest = hashlib.sha256(_prompt_material(build_policy).encode("utf-8")).hexdigest()
    return PromptSpec(
        domain=PROMPT_DOMAIN,
        build_policy=build_policy,
        version=PROMPT_VERSION,
        sha256=digest,
        source=PROMPT_SOURCE,
        approval=PROMPT_APPROVAL,
    )


def build_prompt_manifest(
    build_policy: str,
    *,
    build_model: str | None = None,
    build_temperature: float | None = None,
    build_seed: int | None = None,
    build_top_k: int | None = None,
    trajectory_file: str | None = None,
    trajectory_count: int | None = None,
) -> dict:
    manifest = {
        "schema_version": 2,
        "prompt": get_prompt_spec(build_policy).as_dict(),
    }
    manifest["build_model"] = build_model
    manifest["build_temperature"] = build_temperature
    manifest["build_seed"] = build_seed
    manifest["build_top_k"] = build_top_k
    manifest["trajectory_file"] = trajectory_file
    manifest["trajectory_count"] = trajectory_count
    return manifest


def prompt_manifest_mismatches(manifest: dict, expected: PromptSpec) -> list[str]:
    cached_prompt = manifest.get("prompt", {})
    expected_prompt = expected.as_dict()
    identity_keys = ("domain", "build_policy", "version", "sha256")
    return [
        key
        for key in identity_keys
        if cached_prompt.get(key) != expected_prompt.get(key)
    ]


def generate_workflow_from_trajectory_prompt(query, trajectory):
    prompt = DIRECT_PROMPT_TEMPLATE.format(query=query, trajectory=trajectory)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def generate_events_from_trajectory_prompt(query, trajectory):
    prompt = EVENTS_PROMPT_TEMPLATE.format(query=query, trajectory=trajectory)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def generate_workflow_from_events_prompt(query, events):
    prompt = WORKFLOW_FROM_EVENTS_PROMPT_TEMPLATE.format(query=query, events=events)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
