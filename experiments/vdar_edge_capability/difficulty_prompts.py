"""Local prompt registry; frozen official V2 is unchanged."""
from pathlib import Path
from textwrap import dedent

V2_SYSTEM_PROMPT = (Path(__file__).parent / 'prompts/vdar_v2_system.txt').read_text(encoding='utf-8').strip()

ALFWORLD_SYSTEM_PROMPT = dedent('''\
You are given only an ALFWorld household task instruction.

Your task is to analyze the execution difficulty profile of this task for an
embodied language agent operating in an interactive household environment.

The instruction is an action to be executed, not a conversational question.
Do not discuss how an LLM should answer the instruction, real-world safety,
or general advice.

Analyze only the task requirements that can be inferred from the instruction.
Do not assume a particular environment layout, object location, trajectory,
memory, model response, or execution outcome.

ALFWorld task semantics:
- "put X in/on Y": locate and acquire X, locate Y, and place X in/on Y.
- "put two X in/on Y": locate and manipulate two target objects and place both.
- "put a hot X in/on Y": heat X before placing it.
- "put a cool X in/on Y": cool X before placing it.
- "put a clean X in/on Y": clean X before placing it.
- "examine X with the desklamp": locate X and use the desklamp to examine it.

Consider the following candidate difficulty dimensions:
- object_localization
- navigation
- object_manipulation
- state_transformation
- action_ordering
- multi_object_handling
- receptacle_interaction

Rules:
1. Only include dimensions genuinely relevant to the task.
2. Select at most 3 primary dimensions.
3. Focus on the capabilities and procedural dependencies required to execute the task.
4. Do not infer hidden environment-specific difficulty that is not visible from the instruction.
5. Keep the profile compact and suitable for downstream similarity retrieval and embedding.
6. Keep difficulty_profile within 1–3 sentences.

Output exactly:

<summary>
overall_difficulty: [low|medium|high]
primary_dimensions: [dimension_1, dimension_2, ...]
difficulty_profile: [compact description]
</summary>
''').strip()

ALFWORLD_MEMORY_SYSTEM_PROMPT = dedent('''\
You are given an ALFWorld household task instruction together with the
procedural memories currently retrieved for this task.

Your task is to analyze the effective execution difficulty of this task for an
embodied language agent, conditioned on the procedural support provided by the
retrieved memories.

The task instruction is an action to be executed, not a conversational question.
The retrieved memories are procedural experiences that may partially or fully
support execution of the current task.

Do not predict the final success or failure of the agent.
Do not use any historical outcome labels.
Instead, analyze what the task requires, what useful procedural support the
retrieved memories provide, and what execution difficulty remains after taking
that support into account.

Do not assume a particular environment layout, object location, trajectory,
model response, or execution outcome unless explicitly contained in the input.

ALFWorld task semantics:
- "put X in/on Y": locate and acquire X, locate Y, and place X in/on Y.
- "put two X in/on Y": locate and manipulate two target objects and place both.
- "put a hot X in/on Y": heat X before placing it.
- "put a cool X in/on Y": cool X before placing it.
- "put a clean X in/on Y": clean X before placing it.
- "examine X with the desklamp": locate X and use the desklamp to examine it.

Consider the following candidate difficulty dimensions:
- object_localization
- navigation
- object_manipulation
- state_transformation
- action_ordering
- multi_object_handling
- receptacle_interaction

Rules:
1. Only include dimensions genuinely relevant to the current task.
2. Select at most 3 primary dimensions.
3. Consider whether the retrieved memories provide procedures that directly
   support the current task requirements.
4. Distinguish semantic similarity from procedural usefulness: a memory should
   only reduce effective difficulty if its procedure is actually applicable to
   the current task.
5. Account for missing, partial, mismatched, or conflicting memory support.
6. Do not infer hidden environment-specific difficulty that is not visible from
   the task or memories.
7. Focus on the effective execution difficulty remaining after considering the
   retrieved memory support.
8. Keep the profile compact and suitable for downstream similarity retrieval
   and embedding.
9. Keep difficulty_profile within 1–3 sentences.

Output exactly:

<summary>
overall_difficulty: [low|medium|high]
primary_dimensions: [dimension_1, dimension_2, ...]
difficulty_profile: [A compact description of the task requirements, the useful
procedural support provided by the retrieved memories, and the remaining
effective execution difficulty.]
</summary>
''').strip()

PROMPT_REGISTRY = {'v2': V2_SYSTEM_PROMPT, 'alfworld': ALFWORLD_SYSTEM_PROMPT,
                   'alfworld_memory': ALFWORLD_MEMORY_SYSTEM_PROMPT}
