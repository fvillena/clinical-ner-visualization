FROM pytorch/pytorch

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt requirements.txt

# RUN apt-get update --allow-insecure-repositories && apt-get install -y pkg-config git make

RUN pip3 install -r requirements.txt

COPY app.py app.py
COPY clinicalner.py clinicalner.py

EXPOSE 7860

CMD [ "python", "app.py"]