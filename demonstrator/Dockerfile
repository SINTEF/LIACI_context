FROM python:3.9-buster

# Set labels 
LABEL vendor=SINTEF_Digital 

RUN mkdir -p /opt/liaci-context/assets/imgs \
    && mkdir -p /opt/liaci-context/data/access
WORKDIR /opt/liaci-context
ENV PYTHONPATH /opt/liaci-context

RUN apt-get -y update && apt-get install -y curl && apt-get clean

COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

COPY views views
COPY data_manager.py . 
COPY app.py . 
COPY neo4j_datastore.py .
COPY main.sh . 

COPY assets/LiaciContextLogo.png assets/
COPY assets/sintef_logo.png assets/
COPY assets/style.css assets/
COPY assets/favicon.ico assets/


VOLUME [ "/opt/liaci-context/assets/imgs" ]
EXPOSE 8051

ENTRYPOINT ["/bin/bash", "/opt/liaci-context/main.sh" ]
