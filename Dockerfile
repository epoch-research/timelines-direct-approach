FROM ubuntu:22.04
WORKDIR /home

COPY *.py .
COPY *.txt .
COPY data/ ./data
RUN ls -la .
RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN apt-get install -y python3-pip

RUN pip3 install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
