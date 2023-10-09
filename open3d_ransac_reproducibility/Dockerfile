# ARG OPEN3D_VERSION=v0.16.0
# ARG OPEN3D_VERSION=v0.17.0
# ARG OPEN3D_VERSION=master

ARG OPEN3D_VERSION
ARG DOCKER_BASE_IMAGE=ubuntu:focal
FROM ${DOCKER_BASE_IMAGE}

ARG OPEN3D_VERSION=${OPEN3D_VERSION}

# set version label
ARG BUILD_DATE
ARG VERSION
LABEL build_version="version:- ${VERSION} Build-date:- ${BUILD_DATE}"

# global environment settings
ENV DEBIAN_FRONTEND="noninteractive"

WORKDIR /ws

# install bootstrap packages
RUN apt-get update -qq \
    && apt-get -qq install --no-install-recommends -y sudo git build-essential curl wget ca-certificates gpg python3-pip python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip

RUN apt-get update -qq \
    && wget -q -O - https://apt.kitware.com/kitware-archive.sh | bash \
    && apt install -y cmake cmake-curses-gui \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/isl-org/Open3D -b ${OPEN3D_VERSION} --depth 1 \
    && cd Open3D \
    && apt-get update -qq \
    && yes | bash util/install_deps_ubuntu.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p Open3D/build \
    && cd Open3D/build \
    && cmake .. \
    && make -j$(nproc) \
    && make install-pip-package \
    && make pip-package

RUN python -c "import open3d as o3d ; print(o3d.__version__)"

RUN mkdir -p /shared \
    && cp Open3D/build/lib/python_package/pip_package/*.whl /shared

COPY plane_test.py .

RUN python plane_test.py