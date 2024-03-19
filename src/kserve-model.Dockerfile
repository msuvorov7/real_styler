FROM python:3.8-slim

COPY ./src/kserve_model kserve_model

RUN pip install --upgrade pip && pip install -r ./kserve_model/requirements.txt

RUN mkdir -p /mnt/models/
COPY ./models/model_wave.onnx /mnt/models/

RUN useradd kserve -m -u 1000 -d /home/kserve
USER 1000
ENTRYPOINT ["python", "kserve_model/model.py"]