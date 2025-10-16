FROM ghcr.io/astral-sh/uv:0.6-python3.12-bookworm AS base
RUN apt-get update -q -y && apt-get install -y libgl1-mesa-dev && apt-get clean && apt-get autoclean
WORKDIR /converter
COPY ./ /converter
RUN uv pip install --system -e ./[waymo,av2] lanelet2 git+https://github.com/motional/nuplan-devkit.git aioboto3 retry
RUN uv pip install --system nuscenes-devkit && uv pip install --system --upgrade numpy
ENTRYPOINT ["omega-prime", "from-trajdata"]