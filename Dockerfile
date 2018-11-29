FROM nvidia/cuda:9.0-cudnn7-devel

SHELL ["/bin/bash", "-c"]

RUN mkdir /input
RUN mkdir /output
RUN mkdir /license

RUN apt-get update && apt-get install -y wget && apt-get install -y git
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

# Add channels and build, then install
RUN conda create -n perses python=3.6
RUN source activate perses

RUN conda config --add channels omnia
RUN conda update --yes -n base conda
RUN conda config --add channels conda-forge
RUN conda update --yes --all
RUN conda config --add channels omnia/label/dev
RUN conda config --add channels openeye
RUN conda config --add channels omnia/label/rccuda90


RUN conda install --yes conda-build jinja2 anaconda-client pip
RUN pip install awscli

RUN git clone -b experiment-harness https://github.com/choderalab/perses

RUN conda build perses/devtools/conda-recipe
RUN conda install --yes --use-local perses-dev
