1. docker build -t jupyter-ml .
2. docker run --rm -d -t --name=ml -p 8888:8888 --mount src="$(pwd)",target=/usr/src/app,type=bind jupyter-ml
3. Docker exec -ti ml bash