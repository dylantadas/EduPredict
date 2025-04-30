#!/bin/bash

# initialize conda
source /opt/anaconda3/etc/profile.d/conda.sh

# activate edupredict environment
conda activate edupredict

# force environment interpreter
export PATH="/opt/anaconda3/envs/edupredict/bin:$PATH"
echo "Using Python interpreter: $(which python)"