# PDDL Data Structure

This directory contains PDDL domains organized by problem type. Each domain folder should contain:

## Directory Structure:
```
src/data/
├── tetris/
│   ├── README.md            # Domain-specific documentation
│   ├── domain.pddl          # Domain definition in PDDL
│   ├── problem_01.pddl      # Problem instances
│   ├── problem_02.pddl
│   └── ...
├── citycar/
│   ├── README.md            
│   ├── domain.pddl
│   ├── problem_01.pddl
│   └── ...
├── new_problem/
    └── ...
```

## File Naming Conventions:

### Domain Files:
- `domain.pddl` - The main domain definition file
- OR `{domain_name}_domain.pddl` - Alternative naming

### Problem Files:
- `problem_01.pddl`, `problem_02.pddl`, etc.
- `prob01.pddl`, `prob02.pddl`, etc.
- Any `.pddl` file that is NOT the domain file

## Expected Content:

Each domain folder represents a specific planning problem (tetris, city navigation, logistics, etc.) and contains:
1. **README.md file** providing domain-specific documentation and examples
2. **One domain file** defining the planning domain (actions, predicates, types)
3. **Multiple instance files** defining specific instances for that problem

The file manager will automatically discover and organize these files for processing by the LLM planning system.

## Adding New Domains:

1. Create a new folder under `src/data/` with the domain name
2. Add the domain definition file (`domain.pddl`)
3. Add problem instance files (`problem_*.pddl`)
4. The system will automatically detect and process the new domain