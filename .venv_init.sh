#!/bin/bash

# initialize conda
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    export PATH="/opt/anaconda3/bin:$PATH"
fi
unset __conda_setup

# activate edupredict environment
conda activate edupredict

# force environment interpreter
export PATH="/opt/anaconda3/envs/edupredict/bin:$PATH"
echo "Using Python interpreter: $(which python)"