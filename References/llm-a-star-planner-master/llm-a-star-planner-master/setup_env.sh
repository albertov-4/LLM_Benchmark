#!/bin/bash

# Setup the Conda environment for the project
ENV_NAME="a3i"
YAML_FILE="env.yml"

# Ensure `conda` is initialized
eval "$(conda shell.bash hook)"

# Check if the Conda environment exists
conda env list | grep -q "^$ENV_NAME "

# shellcheck disable=SC2181
if [ $? -eq 0 ]; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  echo "Conda environment '$ENV_NAME' does not exist. Creating it from '$YAML_FILE'..."
  if [ -f "$YAML_FILE" ]; then
    conda env create -f "$YAML_FILE"
    if [ $? -eq 0 ]; then
      echo "Environment '$ENV_NAME' created successfully."
    else
      echo "Error: Failed to create the environment. Check the YAML file."
      exit 1
    fi
  else
    echo "Error: YAML file '$YAML_FILE' not found."
    exit 1
  fi
fi

# Output the command to activate the environment in the parent shell
echo "Run the following command to activate the environment in your shell:"
echo "conda activate $ENV_NAME"
