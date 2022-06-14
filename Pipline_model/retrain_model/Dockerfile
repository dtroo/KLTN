FROM python:3.7-slim

WORKDIR /app

#RUN pip install -U scikit-learn numpy

COPY retrain.py ./retrain.py

COPY requirements.txt ./requirements.txt

RUN pip install tensorflow tensorflow-gpu 

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "retrain.py" ]