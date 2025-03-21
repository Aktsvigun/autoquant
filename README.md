## Auto quantization for TeamCity integration

### Script Arguments

- **`--config-path`**: Path to the configuration file. Default is `config.json`.
- **`--model-id`**: Model ID on HuggingFace (e.g., `Qwen/Qwen2.5-VL-7B-Instruct`).
- **`--model-type`**: Class from HuggingFace's `transformers` library used to load the model. Found in the model's card under "Use with transformers" (e.g., `Qwen2_5_VLForConditionalGeneration`).
- **`--task`**: Task for which the model is used. Supported values: `"text-to-text"`, `"image-to-text"`.
- **`--ports`**: Comma-separated ports for serving the original and quantized models (e.g., `8001,8002`).
- **`--num-gpus-first-model`**: Number of GPUs for the original model. Use `-1` for automatic GPU allocation.
- **`--num-concurrent`**: Number of concurrent requests to send to each server.
- **`--max-tokens`**: Maximum number of tokens to generate for all tasks.
- **`--do-quantize`**: Flag to enable quantization and model saving.
- **`--ignore-modules`**: List of modules to ignore during quantization (e.g., `["lm_head"]` or `["re:.*lm_head", "re:visual.*"]`). Refer to [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main/examples) examples or consult @aktsvigun for guidance.

These arguments can be specified either in the `config.json` file or passed via the command line using `ArgParser`. Command-line arguments take precedence over those in the configuration file.

### Create the Docker image: 
```bash
docker build -t quant-eval .
```

### Launch the container (example):
```bash
docker run -it --gpus all -v ./model-storage:/app/model-storage   -v ./eval-generations:/app/eval-generations -v ./cache:/app/cache   -e NEBIUS_API_KEY=$NEBIUS_API_KEY    -e HF_TOKEN=$HF_TOKEN   -p 8007-8008:8007-8008   quant-eval
```

### Example Commands

#### Text-to-Text Model
You can run a text-to-text model evaluation using either command-line arguments or a config file:

Using command-line arguments:
```bash
python quantize_and_eval.py --model-id "Qwen/Qwen2.5-7B-Instruct" \
               --model-type "AutoModelForCausalLM" \
               --task "text-to-text" \
               --ports "8007,8008" \
               --num-gpus-first-model 1 \
               --num-concurrent 16 \
               --max-tokens 2048 \
               --do-quantize \
               --ignore-modules '["lm_head"]'
```

Using a config file:
```bash
# Example config.json for text-to-text model
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "model_type": "AutoModelForCausalLM",
  "task": "text-to-text",
  "ports": "8007,8008",
  "num_gpus_first_model": 1,
  "num_concurrent": 16,
  "max_tokens": 2048,
  "do_quantize": true,
  "ignore_modules": ["lm_head"]
}

# Run with config file
python quantize_and_eval.py --config-path "config.json"
```

#### Image-to-Text Model
For image-to-text models, you need to use a vision-language model:

Using command-line arguments:
```bash
python quantize_and_eval.py --model-id "Qwen/Qwen2.5-VL-72B-Instruct" \
               --model-type "Qwen2_5_VLForConditionalGeneration" \
               --task "image-to-text" \
               --ports "8007,8008" \
               --num-gpus-first-model 4 \
               --num-concurrent 16 \
               --max-tokens 2048 \
               --do-quantize \
               --ignore-modules '["lm_head", "re:visual.*"]'
```

Using a config file:
```bash
# Example config.json for image-to-text model
{
  "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
  "model_type": "Qwen2_5_VLForConditionalGeneration",
  "task": "image-to-text",
  "ports": "8007,8008",
  "num_gpus_first_model": 4,
  "num_concurrent": 16,
  "max_tokens": 2048,
  "do_quantize": true,
  "ignore_modules": ["lm_head", "re:visual.*"]
}

# Run with config file
python quantize_and_eval.py --config-path "config.json"
```

### Output example
The script will output whether the quantized model has a statistically significant quality drop compared to that of the original model.

Example output when there is no significant difference:
```bash
2025-02-26 09:42:21,403 - INFO - ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Paired t-test results ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-26 09:42:21,403 - INFO - --------------- Dataset: Aktsvigun/nebius_eval_gpqa_diamond ---------------
2025-02-26 09:42:21,406 - INFO - p-value: 1.00000
2025-02-26 09:42:21,406 - INFO - The difference is not statistically significant (p >= 0.05). The quantized model performance is similar.
2025-02-26 09:42:21,407 - INFO - --------------- Dataset: Aktsvigun/nebius_eval_mmlu_pro ---------------
2025-02-26 09:42:21,408 - INFO - p-value: 0.37911
2025-02-26 09:42:21,408 - INFO - The difference is not statistically significant (p >= 0.05). The quantized model performance is similar.
2025-02-26 09:42:21,412 - INFO - Quantized model doesn't have a quality drop
```

Example output when the difference is significant:
```bash
2025-02-26 09:42:21,403 - INFO - ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Paired t-test results ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-26 09:42:21,403 - INFO - --------------- Dataset: Aktsvigun/nebius_eval_gpqa_diamond ---------------
2025-02-26 09:42:21,406 - INFO - p-value: 0.00081
2025-02-26 09:42:21,406 - INFO - The difference is statistically significant (p < 0.05). The quantized model likely has lower performance.
2025-02-26 09:42:21,407 - INFO - --------------- Dataset: Aktsvigun/nebius_eval_mmlu_pro ---------------
2025-02-26 09:42:21,408 - INFO - p-value: 0.37911
2025-02-26 09:42:21,408 - INFO - The difference is not statistically significant (p >= 0.05). The quantized model performance is similar.
2025-02-26 09:42:21,412 - INFO - Quantized model has a quality drop
```