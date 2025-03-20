# anaconda as base image
FROM continuumio/anaconda3

# set wd inside container
WORKDIR /workspace

# install dependencies
RUN conda install -n base -c conda-forge jupyterlab numpy pandas matplotlib seaborn scikit-learn

# create and activate environment
RUN conda create -n myenv python=3.10 && \
    echo "conda activate myenv" >> ~/.bashrc

# jupyter server port
EXPOSE 8888

# command on startup
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]