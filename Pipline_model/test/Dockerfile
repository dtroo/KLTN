FROM python:3.7-slim

WORKDIR /app

#RUN pip install -U scikit-learn numpy

COPY test.py ./test.py

COPY requirements.txt ./requirements.txt

#RUN pip install tensorflow tensorflow-gpu

#RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "test.py" ]