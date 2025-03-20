# anaconda as base image
FROM continuumio/anaconda3

# set working directory
WORKDIR /workspace

# create codespace user and group (UID 1000 default)
RUN groupadd -r codespace && \
    useradd -m -r -g codespace -u 1000 codespace && \
    echo "codespace:codespace" | chpasswd && \
    adduser codespace sudo

# assign ownership and switch to codespace user
RUN chown -R codespace:codespace /workspace

# install common libraries
RUN conda install -n base -c conda-forge jupyterlab numpy pandas matplotlib seaborn scikit-learn

# create and activate conda environment
RUN conda create -n edupredenv python=3.10 && \
    echo "conda activate edupredenv" >> /home/codespace/.bashrc && \
    echo "source activate edupredenv" >> /home/codespace/.bashrc

# assure user has proper permissions
RUN chown -R codespace:codespace /home/codespace

# force environment activation by default
ENV PATH="/opt/conda/envs/edupredenv/bin:$PATH"
ENV CONDA_DEFAULT_ENV="edupredenv"

# jupyter server port
EXPOSE 8888

# switch to codespace user
USER codespace

# start container with environment activated
CMD ["/bin/bash", "-c", "source ~/.bashrc && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]