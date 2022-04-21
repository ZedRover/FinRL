#!/bin/bash
 
 docker run -d --restart=always -v "${PWD}":/home -p 9001:8888 finrl