#!/bin/bash
docker build --build-arg="TARGET=pea" -t ginan-argo -f ./docker/Dockerfile .