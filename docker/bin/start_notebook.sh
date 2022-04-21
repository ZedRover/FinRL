#!/bin/bash

docker run -it --rm -v "${PWD}":/home -p 9001:8888 finrl
