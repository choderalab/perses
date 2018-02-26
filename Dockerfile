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
RUN conda config --add channels omnia
RUN conda config --add channels omnia/label/dev
RUN conda install --yes conda-build==2.1.17 jinja2 anaconda-client pip
RUN conda build devtools/conda-recipe
RUN conda install --yes --use-local perses-dev
RUN pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits
