# Prompts

This folder contains the text files used to build model messages.

Prompts do not select the model, protocol or task. They only define how a task
is presented to the model.

## Structure

```text
prompts/
|-- system.txt
|-- <task_family>.txt
|-- feedback.txt
|-- examples/
    |-- <task_family>.txt
```

## `system.txt`

Global benchmark prompt.

It is included as a `system` message when the protocol has:

```yaml
prompting:
  use_system_prompt: true
```

It defines rules shared across domains:

- use only actions defined in the domain
- use only objects present in the problem
- do not invent predicates, objects or numeric fluents
- produce grounded PDDL actions
- respect preconditions, effects and the final goal

This file should stay general and should not contain task-family-specific
details.

## `<task_family>.txt`

Task-family-specific prompt.

Example:

```text
prompts/<task_family>.txt
```

It is loaded when the protocol has:

```yaml
prompting:
  include_domain_prompt: true
```

This file is required. If `prompts/<task_family>.txt` is missing, the benchmark
fails before calling the model.

There is no automatic fallback to `default.txt`. This prevents ambiguous runs
where a task family is evaluated with a generic prompt instead of its own
domain-specific instructions.

The prompt should explain:

- the domain semantics
- important constraints
- available actions
- common mistakes to avoid
- expected formatting, when domain-specific formatting matters

## `examples/<task_family>.txt`

Optional examples for one task family.

Example:

```text
prompts/examples/<task_family>.txt
```

It is loaded only when the protocol has:

```yaml
prompting:
  include_examples: true
```

Examples clarify action formatting and common mistakes. They should not solve
benchmark instances.

## `feedback.txt`

Prompt used by repair protocols.

It is included in feedback messages when the protocol has:

```yaml
prompting:
  include_external_feedback: true
```

The runner combines this text with validator output. The model receives:

- general repair rules
- validation status
- error type
- failed step, when available
- failed action, when available
- validator feedback

The model should return a complete corrected plan, not only a patch or the
single failed action.

## Protocol Flow

`direct_plan`

```text
system.txt
+ <task_family>.txt
+ dominio PDDL
+ problema PDDL
```

The model should produce the final plan directly.

`direct_plan_with_rationale`

```text
system.txt
+ <task_family>.txt
+ examples/<task_family>.txt
+ dominio PDDL
+ problema PDDL
```

The model may produce a short rationale if the protocol does not require
plan-only output. The final plan must still be extractable as PDDL actions.

`iterative_repair`

First attempt:

```text
system.txt
+ <task_family>.txt
+ examples/<task_family>.txt
+ dominio PDDL
+ problema PDDL
```

Later attempts:

```text
previous messages
+ feedback.txt
+ validator feedback
```

Each iteration is saved in the output `attempts` field, including the messages
sent to the model.

## Adding A New Task Family

To add a new task family, create at least:

```text
tasks/<task_family>/domain/domain.pddl
tasks/<task_family>/easy/<instance_id>.pddl
prompts/<task_family>.txt
```

If the protocol uses examples, also add:

```text
prompts/examples/<task_family>.txt
```

Without `prompts/<task_family>.txt`, every protocol with
`include_domain_prompt: true` will fail before querying the model.
