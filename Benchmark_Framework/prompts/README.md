# Prompts

This folder contains text fragments used to build model messages. Prompts do
not choose models, protocols, or tasks; they only control how a selected task is
presented to the selected model.

## Layout

```text
prompts/
|-- system.txt
|-- feedback.txt
|-- <task_family>.txt
`-- examples/
    `-- <task_family>.txt
```

There are prompt files for the current task families:

- `block-grouping`
- `expedition`
- `fo-counters`
- `fo-sailing`
- `rover`
- `settlersnumeric`

## `system.txt`

The global system prompt is included when a protocol sets:

```yaml
prompting:
  use_system_prompt: true
```

It contains benchmark-wide rules: use only domain actions, use only problem
objects, do not invent predicates or numeric fluents, produce grounded PDDL
actions, and respect preconditions, effects, and goals.

Keep this file general. Task-family-specific semantics belong in
`<task_family>.txt`.

## `<task_family>.txt`

Task-family prompts are required when a protocol sets:

```yaml
prompting:
  include_domain_prompt: true
```

The runner does not fall back to a generic prompt. Missing task-family prompts
fail before a model call so that benchmark runs are not silently evaluated with
ambiguous instructions.

Each task-family prompt should explain:

- domain semantics;
- important constraints;
- available actions and naming conventions;
- common mistakes to avoid;
- formatting expectations that are specific to the domain.

## `examples/<task_family>.txt`

Example prompts are optional and loaded only when:

```yaml
prompting:
  include_examples: true
```

Examples are appended after the task-family prompt. They should clarify action
format and constraints, but should not solve benchmark instances. If an example
file is missing, the benchmark continues without examples.

## `feedback.txt`

The repair prompt is used by protocols with:

```yaml
prompting:
  include_external_feedback: true
```

The runner combines this text with validator output, including status, error
type, failed step, failed action, and validator feedback when available. If the
provider reasoning contains a decoded valid plan while the raw final answer does
not, the repair message can include that decoded action sequence as a hint. The
model is asked to return a complete corrected plan.

## Protocol Assembly

`direct_plan`:

```text
system.txt
+ <task_family>.txt
+ domain PDDL
+ problem PDDL
```

`direct_plan_with_rationale`:

```text
system.txt
+ <task_family>.txt
+ examples/<task_family>.txt
+ domain PDDL
+ problem PDDL
```

`iterative_repair`, first attempt:

```text
system.txt
+ <task_family>.txt
+ examples/<task_family>.txt
+ domain PDDL
+ problem PDDL
```

`iterative_repair`, later attempts:

```text
previous messages
+ feedback.txt
+ normalized validator feedback
```

Each attempt is saved in the output artifacts.

Repair feedback may include raw parse issues such as unknown actions, wrong
arity, reasoning mixed into the final answer, or missing domain-valid actions.
It must still ask for a complete corrected sequence of PDDL actions, one per
line. Provider-side reasoning is not official scoring input, but a decoded valid
reasoning plan may be echoed as a repair hint when raw output fails.
