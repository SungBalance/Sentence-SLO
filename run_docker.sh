#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${PWD}"
CACHE_DIR="/data"
CONTAINER_WORKDIR="/workspace/mlsys"
CONTAINER_CACHE_DIR="/cache"

COMMON_DOCKER_ARGS=(
    -d
    -it
    -v "${REPO_DIR}:${CONTAINER_WORKDIR}"
    -v "${CACHE_DIR}:${CONTAINER_CACHE_DIR}"
    --ipc=host
    --runtime=nvidia
    --gpus all
    --cap-add=SYS_ADMIN
    --ulimit memlock=-1
    --restart=unless-stopped
    --privileged
    --network=host
)

if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
else
    DOCKER=(sudo docker)
fi

start_container() {
    local name="$1"
    local image="$2"

    "${DOCKER[@]}" run "${COMMON_DOCKER_ARGS[@]}" \
        --name "${name}" \
        "${image}" bash
}

case "${1:-vllm}" in
    vllm)
        start_container "sk-sslo" "nvcr.io/nvidia/vllm:26.03-py3"
        ;;
    omni|omni-0.18|omni-0.18.0)
        start_container "sk-sslo-omni" "vllm/vllm-omni:v0.18.0"
        ;;
    omni-0.16|omni-0.16.0)
        start_container "sk-sslo-omni-016" "vllm/vllm-omni:v0.16.0"
        ;;
    *)
        echo "Usage: $0 [vllm|omni|omni-0.18|omni-0.16]" >&2
        exit 1
        ;;
esac
