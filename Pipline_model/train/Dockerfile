FROM python:3.7-slim

WORKDIR /app

#RUN pip install -U scikit-learn numpy

COPY train.py ./train.py

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "train.py" ]