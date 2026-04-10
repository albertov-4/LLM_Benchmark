# A3I Project - Choreography Planning by Decoupling A* Deterministic Search and Heuristic Computation

## Overview
This project uses advanced techniques to design and evaluate dance choreographies for the NAO robot. By combining heuristic methods, a custom A* search algorithm, and large language model (LLM) evaluations, the system generates creative and engaging robot movements tailored to specific constraints and criteria.

## Features
- **Choreography Evaluation**: Rates choreography based on storytelling, rhythm, movement technique, public involvement, space use, human characterization, and human reproducibility.
- **A* Search Algorithm**: Custom implementation to find the optimal sequence of dance moves while adhering to constraints like duration and mandatory moves.
- **Heuristic Computation**: Combines past scores (g-function) and future estimations (h-function) to guide the search process.
- **Dynamic Scoring**: Allows adjusting weights for different scoring components (e.g., waypoint adherence, duration bonus).

## Components
1. **LLM Scoring Functions**:
   - `g_llm`: Evaluates the coolness of a complete dance sequence.
   - `h_llm`: Estimates the coolness of potential moves for an incomplete sequence.
2. **Custom A* Implementation**:
   - Uses LLM-generated scores to find optimal paths for robot choreography.
   - Incorporates dynamic constraint satisfaction.
3. **Scoring Criteria**:
   - `gh_weight`: Importance of coolness scores.
   - `t_weight`: Importance of adhering to waypoints.
   - `db_weight`: Bonus for optimal duration.
   - `csc_weight`: Importance of satisfying constraints.

## Input Data
- **Initial State**: Starting position of the robot.
- **Final State**: Target position to complete the choreography.
- **Waypoints**: Intermediate positions the robot must pass through.
- **Mandatory Positions**: Specific moves required in the choreography.
- **Intermediate Positions**: Optional moves for enhanced creativity.
- **Graph**: A connectivity graph representing valid transitions between moves.

## Outputs
- A complete sequence of moves.
- Coolness scores for the choreography.
- Computation cost and number of LLM calls.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/lorenzobalzani/llm-a-star-planner.git
   ```
2. Make sure to have conda installed onto your system.
3. Set up the conda environment by launching the script `./setup_env.sh` in the terminal.
4. Activate it with `conda activate a3i`.
5. For installing the ipython kernel, please run `python -m ipykernel install --user --name=a3i`.

## Setup
Create a file named `secrets.env` in the root directory of the project and add the following lines:
```bash
AZURE_OPENAI_ENDPOINT = <your-deployment-edpoint>
AZURE_OPENAI_API_KEY = <your-api-key>
OPENAI_API_VERSION = <your-api-version>
```
If you prefer to use an LLM other than Azure OpenAI, update the LLM client definitions in the notebook accordingly

## Example Output
```
Choreography: ['INITIAL_stand_init', 'mandatory_stand', 'diagonal_left', ... 'FINAL_crouch']
Cost: 0.06451€
Number of calls to LLM: 59
Elapsed time: 31.50s
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Authors
- **Davide Bombardi** (<a href="mailto:davide.bombardi@studio.unibo.it">davide.bombardi@studio.unibo.it</a>)
- **Lorenzo Balzani** (<a href="mailto:lorenzo.balzani@studio.unibo.it">lorenzo.balzani@studio.unibo.it</a>)

## License
This project is licensed under the MIT License.

