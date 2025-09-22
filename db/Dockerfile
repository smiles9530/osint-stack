FROM postgis/postgis:16-3.4
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release wget apt-transport-https \
    && rm -rf /var/lib/apt/lists/*
RUN sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -cs) main' > /etc/apt/sources.list.d/timescaledb.list" \
    && curl -L https://packagecloud.io/timescale/timescaledb/gpgkey | apt-key add -
RUN curl https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get update && apt-get install -y --no-install-recommends \
    timescaledb-2-postgresql-16 \
    postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*
USER postgres
