# Farmland

This task family contains PDDL planning problems based on movement between
farms and numeric constraints over available resources.

Structure:
- `domain/domain.pddl`: domain actions, predicates and numeric constraints
- `easy/`: low-complexity instances for quick checks
- `medium/`: intermediate instances
- `hard/`: harder instances for stress-testing planning and repair

Notes:
- instances should remain comparable across models
- each `.pddl` file represents one benchmark case
- difficulty criteria should be documented here when new instances are added
