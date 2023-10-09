#!/bin/bash

docker build . -t build_open3d_image
container_id=$(docker create build_open3d_image --name build_open3d_container)
docker cp ${container_id}:/shared/ .