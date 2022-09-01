FROM sciencedesk/miniconda3-py37-4-8-2
WORKDIR /sdesk-conda

RUN  apt-get update &&  apt-get install -y libgl1-mesa-dev \
gcc \
g++ \
libboost-all-dev \
swig \
&& rm -rf /var/lib/apt/*

# gcc, g++, libboost-all-dev and swig used for python package xylib

COPY trained_models/ /root/trained_models

COPY ./conda_environment.yml /sdesk-conda/conda_environment.yml
RUN conda env create -n sdesk -f /sdesk-conda/conda_environment.yml && \
    conda clean --all --yes

ENV PATH /opt/conda/envs/env/bin:$PATH

SHELL ["conda", "run", "-n", "sdesk", "/bin/bash", "-c"]

# Non-conda packages   .
COPY requirements.txt  /sdesk-conda/requirements.txt
RUN pip install -r requirements.txt

COPY src/ /root/code/
COPY DNN4XPS.ipynb /root/DNN4XPS.ipynb


COPY entrypoint.sh /sdesk-conda/entrypoint.sh

ENTRYPOINT ["/sdesk-conda/entrypoint.sh"]
