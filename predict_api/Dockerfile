FROM python:3.10-slim
WORKDIR /app
COPY predict_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY predict_api /app/predict_api
COPY exported_model /app/exported_model
ENV MODEL_PATH=/app/exported_model
EXPOSE 8000
CMD ["uvicorn", "predict_api.app:app", "--host", "0.0.0.0", "--port", "9000"]
