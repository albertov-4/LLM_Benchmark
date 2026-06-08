# Task Family Template

Use this folder as a starting point when creating a new planning domain.

Checklist:

- copy `_template_domain` and rename it to the new task-family name;
- replace the placeholder `domain/domain.pddl`;
- create `.pddl` instances in `easy`, `medium`, and `hard`;
- add `prompts/<task_family>.txt`;
- add `prompts/examples/<task_family>.txt` if example-enabled protocols should
  use examples for the family;
- keep instance naming consistent across tiers;
- document family-specific assumptions in `tasks/README.md` or in a
  task-family README if the domain needs more detail than the shared inventory.
