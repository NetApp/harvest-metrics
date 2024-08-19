FROM python:3.11  AS build-env

ARG version=3.11
ARG APP=app

COPY . ./$APP

WORKDIR /$APP

COPY requirements.lock /$APP/
RUN sed '/-e/d' requirements.lock > requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install -r ./requirements.txt

FROM gcr.io/distroless/python3-debian12:debug
ARG version=3.11
ARG APP=app

COPY --from=build-env /$APP /$APP
COPY --from=build-env /usr/local/lib/python${version}/site-packages /usr/local/lib/python${version}/site-packages

WORKDIR /$APP
ENV PYTHONPATH=/usr/local/lib/python${version}/site-packages
ENTRYPOINT [ "python", "src/harvest_planner/main.py" ]