# One image for every role — master, worker, driver and the pandas baselines — so
# that "same code, same libraries, different execution model" is literally true.
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless procps curl && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DATA_ROOT=/data

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# pyspark ships the full Spark distribution, so the master/worker daemons come from
# the same pip install as the client library. No separate Spark tarball to version-drift.
ENV SPARK_HOME=/usr/local/lib/python3.10/site-packages/pyspark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

WORKDIR /app
COPY src /app/src
