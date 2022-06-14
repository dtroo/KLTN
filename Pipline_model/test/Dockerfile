#FROM python:3.7-slim
FROM d1v1n3/train-pipline-dtro-2

WORKDIR /app

#RUN pip install -U scikit-learn numpy

COPY test.py ./test.py

COPY requirements.txt ./requirements.txt

#RUN pip install tensorflow tensorflow-gpu

#RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "test.py" ]