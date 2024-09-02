FROM orthancteam/orthanc:latest

WORKDIR /etc/orthanc

COPY orthanc.json /etc/orthanc/orthanc.json

RUN mkdir -p /var/lib/orthanc

EXPOSE 4242 8042

CMD ["orthanc", "/etc/orthanc/orthanc.json"]