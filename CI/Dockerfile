FROM continuumio/anaconda3:latest

# enable shell for conda
SHELL ["/bin/bash", "--login", "-c"]
RUN conda init bash

# copy files
 COPY ./ ./

# create conda env
RUN conda create -n package_env python=3.6.2

# install pytest
RUN conda activate package_env && pip install --user pytest

# install package
RUN conda activate package_env && pip install -e .

# run tests
RUN conda activate package_env && python -m pytest --log-cli-level=INFO tests
