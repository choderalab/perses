FROM mambaorg/micromamba:latest

LABEL org.opencontainers.image.source = "https://github.com/choderalab/perses"
LABEL org.opencontainers.image.description="Experiments with expanded ensemble simulation to explore chemical and mutational space."
LABEL org.opencontainers.image.authors = "John D. Chodera"
LABEL org.opencontainers.image.licenses=MIT
# perses version we want to build
ARG VERSION

# install ps
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Don't buffer stdout & stderr streams, so if there is a crash no partial buffer output is lost
# https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED=1

RUN micromamba install -y -n base -c conda-forge -c openeye openeye-toolkits git pytest-xdist pytest-attrib mpiplus mpich mpi mpi4py clusterutils  "perses==$VERSION" cudatoolkit=11.2 && \
    micromamba clean --all --yes

# Make directory and tell openeye where to find
# license file
USER root
RUN mkdir /openeye
USER $MAMBA_USER
ENV OE_LICENSE=/openeye/oe_license.txt

# Ensure that conda environment is automatically activated
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
