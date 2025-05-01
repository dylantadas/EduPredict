#!/bin/zsh

# initialize conda
source "/opt/anaconda3/etc/profile.d/conda.sh"

# activate edupredict environment
conda activate edupredict

# force environment interpreter
export PATH="/opt/anaconda3/envs/edupredict/bin:$PATH"

# unalias python if it exists
unalias python 2>/dev/null || true

# verify the interpreter
echo "Using Python interpreter: $(which python)"