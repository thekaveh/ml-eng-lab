# Multi-project ML Repository

This repository contains multiple mini ML projects I have worked on.

Each mini-project is self contained within its own folder named with the format `[taks]-[dataset]-[model]-[framework]`. The `image_classification-mnist-ffnn-pytorch`, for example, means:
- `task=image_classification`
- `dataset=mnist`
- `model=ffnn (feed forward neural network)`
- `framework=pytorch`

## Prerequisites

To build and run the projects, you'll need:

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Visual Studio Code: [Install VS Code](https://code.visualstudio.com/download/)
- Remote - Containers extension for VS Code: [Install the extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)


## Getting Started
Assuming:
- `[local-repo-path]`: path to local version of this repo
- `[container-image-name]`: name of container image built off of `Dockerfile` in this repo
- `[container-instance-name]`: name of container instance run off of `[container-image-name]`

Steps:
1. `$ git clone https://github.com/thekaveh/ml [local-repo-path] && cd [local-repo-path]`
2. `$ docker build -t [container-image-name]`
3. `$ docker run --shm-size=4g --rm -d -t --name=[container-instance-name] -p 8888:8888 --mount src="$(pwd)",target=/usr/src/app,type=bind [container-image-name]`
4. `$ code .` > Remote Explorer > Dev Containers > [container-image-name] > Right CLick > Attach to Container
5. (optional)`$ docker exec -ti [container-image-name] bash`