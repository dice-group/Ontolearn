#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="examples/retrieval_eval_under_incomplete.py"
path_kg="KGs/Carcinogenesis/carcinogenesis.owl"

# Define the number of incomplete graphs
NUMBER_OF_INCOMPLETE_GRAPHS=5

# Define the list of levels of incompleteness
LEVELS_OF_INCOMPLETENESS=("0.8" "0.9")
# LEVELS_OF_INCOMPLETENESS=("0.1")

# Iterate over each level of incompleteness
for LEVEL in "${LEVELS_OF_INCOMPLETENESS[@]}"; do
    echo "Running with level_of_incompleteness=$LEVEL..."
    python $PYTHON_SCRIPT --number_of_incomplete_graphs $NUMBER_OF_INCOMPLETE_GRAPHS --level_of_incompleteness $LEVEL --path_kg $path_kg
    echo "Completed with level_of_incompleteness=$LEVEL."
done

echo "All tasks completed."
