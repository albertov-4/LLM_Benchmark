# Prompt Examples

This folder contains optional examples loaded by protocols that set `include_examples: true`.

Convention:
- examples are stored as `examples/<task_family>.txt`
- example content is appended after `prompts/<task_family>.txt`
- examples should clarify expected format and constraints
- examples should not solve benchmark instances
- if an examples file is missing, the benchmark continues without examples

Example path:

```text
prompts/examples/<task_family>.txt
```

The main task-family prompt remains required when a protocol has `include_domain_prompt: true`. Examples are optional.
