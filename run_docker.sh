sudo docker run -d -it \
    -v ${PWD}:/workspace/mlsys \
    -v /data:/cache \
    --name sk-sslo \
    --ipc=host \
    --runtime=nvidia \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --ulimit memlock=-1 \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    nvcr.io/nvidia/vllm:26.03-py3 bash
