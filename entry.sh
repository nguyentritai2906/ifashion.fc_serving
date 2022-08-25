#!/bin/bash

echo "Activate env"
source /venv/bin/activate

echo "Create subprocess"
tensorflow_model_server --port=8500 --rest_api_port=8501 \
    --model_name=fashion-compatibility \
    --model_base_path=/fashion_batch1/serving \
    --enable_batching=true \
    --batching_parameters_file=/fashion_batch1/batching_parameters.txt &

echo "Start server"
python serve.py worker -l INFO --without-web &
tail -f /dev/null
