FROM python:3.7-slim

WORKDIR /app

#RUN pip install -U scikit-learn numpy

COPY preprocess.py ./preprocess.py

COPY goodrequests.txt ./goodrequests.txt

COPY badrequests.txt ./badrequests.txt

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "preprocess.py" ]