FROM python:3.6.8-slim

RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install openslide-tools gcc zlib1g -y \
  && apt-get install -y unzip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-pipeline.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-dependencies -r requirements.txt

RUN apt-get update -y && apt-get install -y libgl1-mesa-dev

COPY scripts scripts
COPY Snakefile Snakefile
COPY pipeline.sh run-slide.sh
COPY run-folder.sh run-folder.sh
COPY test-packages.py test-packages.py

RUN cp /app/run-slide.sh /app/run-slide
RUN cp /app/run-folder.sh /app/run-folder
RUN chmod +x /app/run-slide
RUN chmod +x /app/run-folder

ENV CKPT_URL https://figshare.com/ndownloader/articles/19164209?private_link=f331c773f4a0770db5da

RUN python3 test-packages.py

