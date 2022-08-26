FROM neo4j:4.4.9
RUN apt-get update; apt-get install -y curl
COPY --chmod=0300 ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]