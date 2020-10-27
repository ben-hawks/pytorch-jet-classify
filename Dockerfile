FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER

ENV USER=${NB_USER}

RUN set -x \
    && pip install llvmlite --ignore-installed \
    && pip install torchsummaryX 


RUN set -x \
    && fix-permissions /home/$NB_USER
