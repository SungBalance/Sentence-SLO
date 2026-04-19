#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sslo-pt2512}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$PWD}"
CACHE_DIR="${CACHE_DIR:-/data}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' already exists. Remove it or use another CONTAINER_NAME."
  exit 1
fi

docker run -d -it \
  -v "${WORKSPACE_DIR}:/workspace/Sentence-SLO" \
  -v "${CACHE_DIR}:/cache" \
  --name "${CONTAINER_NAME}" \
  --ipc=host \
  --runtime=nvidia \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --ulimit memlock=-1 \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  "${IMAGE}" bash

echo "Started container: ${CONTAINER_NAME}"
echo "Attach with: docker exec -it ${CONTAINER_NAME} bash"
