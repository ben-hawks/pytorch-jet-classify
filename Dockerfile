FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER

ENV USER=${NB_USER}

RUN set -x \
    && pip install llvmlite --ignore-installed \
    && pip install torchsummaryX \
    && pip install git+https://github.com/Xilinx/brevitas.git


RUN set -x \
    && fix-permissions /home/$NB_USER
