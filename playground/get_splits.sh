#!/bin/bash

# Define the base command
BASE_COMMAND="sudo /home/rbertin/miniconda3/envs/fresh_ss/bin/shapesd alignment add --dataset_path \"/shared/datasets/simple_shapes_dataset\" --seed 0 --domain_alignment v 1.0 --domain_alignment t 1.0 --domain_alignment attr 1.0"

# Define the proportions to be used
PROPORTIONS=(0.00005 0.0001 0.0005 0.002 0.005 0.01 1.0)

# Loop through each proportion and create the command
for PROP in "${PROPORTIONS[@]}"; do
  COMMAND="$BASE_COMMAND --domain_alignment attr,v $PROP --domain_alignment attr,t $PROP --domain_alignment t,v $PROP --domain_alignment attr,t,v $PROP"
  echo "$COMMAND"
done
