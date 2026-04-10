# Tetris Domain

This directory contains PDDL files for Tetris-like spatial planning problems.

## Domain Description:
The Tetris domain involves:
- A grid-based workspace
- Different types of pieces (single squares, two-straight pieces, L-shaped pieces)  
- Movement actions to reconfigure pieces
- Spatial constraints and connectivity rules

## Files Expected:
- `domain.pddl` - Tetris domain definition with actions and predicates
- `problem_*.pddl` - Specific Tetris puzzles with different initial and goal configurations

## Problem Characteristics:
- Grid sizes: varying from small (4x4) to larger (8x8) configurations
- Piece types: one_square, two_straight, right_l
- Actions: movement actions with spatial constraints
- Goals: achieve specific piece arrangements

## Usage:
Place your Tetris PDDL files here and the system will automatically process them for LLM planning experiments.