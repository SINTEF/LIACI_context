FROM alpine
RUN apk add --update curl
COPY --chmod=0300 ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]