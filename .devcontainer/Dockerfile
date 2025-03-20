# anaconda as base
FROM continuumio/anaconda3

# set working directory
WORKDIR /workspace

# install libraries in base environment
RUN conda install -y -n base -c conda-forge \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn

# create dedicated environment
RUN conda create -y -n edupredenv python=3.10

# configure conda for terminals
RUN conda init bash

# activate environment when opening terminal
RUN echo "conda activate edupredenv" >> ~/.bashrc

# set default environment and path
ENV PATH="/opt/conda/envs/edupredenv/bin:${PATH}"
ENV CONDA_DEFAULT_ENV="edupredenv"

# jupyter port
EXPOSE 8888

# command runs jupyter lab with container creation
CMD ["/bin/bash", "-c", "source ~/.bashrc && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]