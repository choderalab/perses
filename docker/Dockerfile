FROM mambaorg/micromamba:latest

LABEL org.opencontainers.image.source = "https://github.com/choderalab/perses"
LABEL org.opencontainers.image.description="Experiments with expanded ensemble simulation to explore chemical and mutational space."
LABEL org.opencontainers.image.authors = "John D. Chodera"
LABEL org.opencontainers.image.licenses=MIT
# perses version we want to build
ARG VERSION

# Don't buffer stdout & stderr streams, so if there is a crash no partial buffer output is lost
# https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED=1

RUN micromamba install -y -n base -c openeye openeye-toolkits git "perses==$VERSION" cudatoolkit==11.7 && \
    micromamba clean --all --yes

# Make directory and tell openeye where to find
# license file
RUN mkdir /openeye
ENV OE_LICENSE=/openeye/oe_license.txt

# Ensure that conda environment is automatically activated
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1