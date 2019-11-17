#!/bin/sh
# This script builds and runs docker container with the following functionality
# enabled:
#  * Run the shell as current user inside docker
#  * Uses nvidia-docker if available
#  * Bind-mount project directory as docker user's home
#  * Allow to run X11 apps from docker
#  * Maps Tensorboard and Jupyter ports to the host side
#  * Allows strace debugging inside docker
#  * Allows usb Host usb access to docker apps

CWD=$(cd `dirname $0`; pwd;)

GITHACK=n
MAPSOCKETS=y
DOCKERFILE=$CWD/docker/gpu_shell.Dockerfile

while test -n "$1" ; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [-n|--no-map-sockets]" >&2
      exit 1
      ;;
    -n|--no-map-sockets)
      MAPSOCKETS=n
      ;;
    *)
      DOCKERFILE="$1"
      ;;
  esac
  shift
done

set -e -x

DOCKER_IMGNAME="${USER}-spleeter-`basename $DOCKERFILE .Dockerfile`"

docker build \
  --build-arg=http_proxy=$https_proxy \
  --build-arg=https_proxy=$https_proxy \
  --build-arg=ftp_proxy=$https_proxy \
  -t "$DOCKER_IMGNAME" \
  -f "$DOCKERFILE" "$CWD/docker"


if test "$MAPSOCKETS" = "y"; then
  PORT_TENSORBOARD=`expr 6000 + $UID - 1000`
  PORT_JUPYTER=`expr 8000 + $UID - 1000`

  DOCKER_PORT_ARGS="-p 0.0.0.0:$PORT_TENSORBOARD:6006 -p 0.0.0.0:$PORT_JUPYTER:8888"
  (
  set +x
  echo
  echo "***************************"
  echo "Host Tensorboard port: ${PORT_TENSORBOARD}"
  echo "Host Jupyter port:     ${PORT_JUPYTER}"
  echo "***************************"
  )
fi

# Remap detach from Ctrl-p to Ctrl+e,e
DOCKER_CFG="/tmp/docker-config-$UID"
mkdir "$DOCKER_CFG" 2>/dev/null || true
cat >$DOCKER_CFG/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF

xhost +local: || true
cp "$HOME/.Xauthority" "$CWD/.Xauthority" || true

if which nvidia-docker >/dev/null 2>&1; then
  DOCKER_CMD=nvidia-docker
else
  DOCKER_CMD=docker
fi

${DOCKER_CMD} --config "$DOCKER_CFG" \
  run -it --rm \
  --volume "$CWD:/workspace" \
  --workdir /workspace \
  -e HOST_PERMS="$(id -u):$(id -g)" \
  -e "CI_BUILD_HOME=/workspace" \
  -e "CI_BUILD_USER=$(id -u -n)" \
  -e "CI_BUILD_UID=$(id -u)" \
  -e "CI_BUILD_GROUP=$(id -g -n)" \
  -e "CI_BUILD_GID=$(id -g)" \
  -e "DISPLAY=$DISPLAY" \
  -e "EDITOR=$EDITOR" \
  -e "TERM=$TERM" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ${DOCKER_PORT_ARGS} \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --privileged -v /dev/bus/usb:/dev/bus/usb \
  "$DOCKER_IMGNAME" \
  bash /with_the_same_user.sh bash --login


