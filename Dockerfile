FROM vllm/vllm-openai:v0.7.3

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY quantize_and_eval.py .
COPY config.json .

ENV PYTHONUNBUFFERED=1 \
    MODEL_SAVE_DIR=/app/model-storage \
    RESULTS_SAVE_DIR=/app/eval-generations \
    CACHE_DIR=/app/cache

RUN mkdir -p ${MODEL_SAVE_DIR} ${RESULTS_SAVE_DIR} ${CACHE_DIR}

EXPOSE 8007-8008

ENTRYPOINT ["python3", "quantize_and_eval.py", "--config-path", "/app/config.json"]