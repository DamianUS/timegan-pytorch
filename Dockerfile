FROM nvcr.io/nvidia/pytorch:22.12-py3
ARG USER=docker
ARG UID=1000
ARG GID=1000
ARG PW=docker
ARG COMMANDS_PATH=./commands.sh
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | \
      chpasswd
WORKDIR /timegan-pytorch
ADD . /timegan-pytorch
RUN chmod +x /timegan-pytorch/docker-entrypoint.sh && pip install -r requirements.txt
USER ${UID}:${GID}
CMD ${COMMANDS_PATH}
