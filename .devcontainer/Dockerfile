# anaconda as base image
FROM continuumio/anaconda3

# set working directory
WORKDIR /workspace

# create codespace user and group (uid 1000 default)
RUN groupadd -r codespace && useradd -m -r -g codespace -u 1000 codespace

# give user necessary permissions
RUN chown -R codespace:codespace /workspace

# switch to codespace user
USER codespace

# install dependencies
RUN conda install -n base -c conda-forge jupyterlab numpy pandas matplotlib seaborn scikit-learn

# create and activate conda environment
RUN conda create -n edupredenv python=3.10 && \
    echo "conda activate edupredenv" >> ~/.bashrc

ENV PATH="/opt/conda/envs/edupredenv/bin:$PATH"
ENV CONDA_DEFAULT_ENV="edupredenv"
    

# jupyter server port
EXPOSE 8888

# command on container startup
CMD ["/bin/bash", "-c", "source ~/.bashrc && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]