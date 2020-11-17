FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER

ENV USER=${NB_USER}

RUN set -x \
    && conda install numpy==1.18.4 --force-reinstall \
    && conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch --force-reinstall \
    && pip install llvmlite --ignore-installed \
    && pip install torchsummaryX h5pickle \
    && pip install git+https://github.com/Xilinx/brevitas.git@cdd46d5a6aad2af24ec2e15c87dcee2ce74860d0


RUN set -x \
    && fix-permissions /home/$NB_USER
