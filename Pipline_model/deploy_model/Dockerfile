FROM python:3.7-slim

WORKDIR /app

COPY deploy_model.py ./deploy_model.py

COPY push.py ./push.py

RUN pip install pygithub

ENTRYPOINT [ "python", "deploy_model.py" ]
