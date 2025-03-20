import argparse
import atexit
import json
import logging
import os
import re
import subprocess
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time, sleep
from typing import Literal, Any
import importlib

import numpy as np
import pandas as pd
import requests
from torch import cuda
from datasets import load_dataset, Dataset, concatenate_datasets
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from openai import Client
from scipy import stats
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CACHE_DIR = "./cache"
NUM_CONCURRENT = os.getenv("NUM_CONCURRENT", 16)
MAX_TOKENS = os.getenv("MAX_TOKENS", 2048)
EVAL_DATASETS = os.getenv("EVAL_DATASETS", "gpqa,mmlu_pro")
MODEL_PATH = os.getenv("model_path", None)
SERVER_PORTS = os.getenv("SERVER_PORTS", "8000,8001")
SERVER_MAX_TIMEOUT = 1800
CONFIDENCE_LEVEL = 0.05
MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR", "/home/aktsvigun/model-storage")
RESULTS_SAVE_DIR = os.getenv("RESULTS_SAVE_DIR", "./eval_generations")
CUDA_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(cuda.device_count())])).split(",")
DO_QUANTIZE = os.getenv("DO_QUANTIZE", "True").lower() in ("true", "1", "t", "yes")

SERVER_TASK_SPECIFIC_ARGS = {
    "text-to-text": [],
    "image-to-text": [
        "--limit-mm-per-prompt", "image=8",
    ],
}
DATASET_MAP = {
    "gpqa": "Aktsvigun/nebius_eval_gpqa_diamond",
    "mmlu_pro": "Aktsvigun/nebius_eval_mmlu_pro",
    # "mmlu_pro_full": "Aktsvigun/nebius_mmlu_pro_full",
    "mmmu": "Aktsvigun/nebius_eval_mmmu",
}
TASK_DATASETS_MAP = {
    "text-to-text": (
        "Aktsvigun/nebius_eval_gpqa_diamond",
        "Aktsvigun/nebius_eval_mmlu_pro",
    ),
    "image-to-text": ("Aktsvigun/nebius_eval_mmmu",),
}

HEALTH_CHECK_URL = "http://127.0.0.1:{port}/health"
DEFAULT_ANSWER_EXTRACTION_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct-fast",
    "meta-llama/Llama-3.3-70B-Instruct-fast",
    "microsoft/phi-4",
    "Qwen/QwQ-32B-Preview",
]
ANSWER_EXTRACTION_MODELS = os.getenv(
    "ANSWER_EXTRACTION_MODELS", DEFAULT_ANSWER_EXTRACTION_MODELS
)
MAX_NUM_RETRIES = len(ANSWER_EXTRACTION_MODELS)
NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1"
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", os.getenv("OPENAI_API_KEY", None))
if NEBIUS_API_KEY is None:
    raise ValueError("Please provide `NEBIUS_API_KEY` or `OPENAI_API_KEY`.")

SYSTEM_PROMPT_EXTRACT_ANSWER = f"""
You will be provided with model's solution of the task. Extract the final answer from the solution. It must be just one letter.
If the answer is not stated clearly or is vague, output "-". Otherwise, output the letter, which is stated as the answer.
""".strip()


def wait_server(port: int) -> bool:
    start_ts = time()
    while time() - start_ts < SERVER_MAX_TIMEOUT:
        health_url = str(HEALTH_CHECK_URL).format(port=port)
        try:
            if requests.get(health_url, timeout=1).status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            pass
        sleep(1)
    return False


def _extract_answer(input, client):
    num_retries = 0
    while num_retries < MAX_NUM_RETRIES:
        try:
            return _call_model(input, client, ANSWER_EXTRACTION_MODELS[num_retries])
        except:
            num_retries += 1
            continue
    return "-"


def _call_model(input, client, model):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXTRACT_ANSWER},
                {"role": "user", "content": input},
            ],
            top_p=0.01,
            max_tokens=1,
            extra_body={"guided_choice": list("ABCDEFGHIJKLMOPQRSTUVWXYZ-")},
        )
        return "(" + completion.choices[0].message.content + ")"
    except Exception as e:
        logger.error(f"Model call failed for model {model}: {e}")


def process_example(
    task: Literal["text-to-text", "image-to-text"],
    messages,
    client,
    max_tokens: int = MAX_TOKENS,
):
    if task == "image-to-text":
        messages = eval(messages)
    try:
        output = client.chat.completions.create(
            model="model",
            top_p=0.01,
            max_tokens=max_tokens,
            messages=messages,
        )
        return output.choices[0].message.content
    except Exception as e:
        logger.error(
            f"Model call failed for model on port {client.base_url.raw.port} with exception: {e}\n"
        )
        import sys, pdb

        exc_type, exc_value, exc_traceback = sys.exc_info()
        pdb.post_mortem(exc_traceback)


def _assign_gpus(num_gpus_first_model):
    num_gpus = len([d for d in CUDA_DEVICES if d])
    if num_gpus_first_model in ("", "-1", -1):
        num_gpus_first_model = num_gpus // 2 + num_gpus % 2

    first_model_gpus = CUDA_DEVICES[:num_gpus_first_model]
    second_model_gpus = CUDA_DEVICES[num_gpus_first_model:]
    # Qwen models require the number of GPUs to be a multiple of 4
    while (not len(second_model_gpus) % 4 == 0) and (len(second_model_gpus) > 1):
        second_model_gpus = second_model_gpus[:-1]
    return first_model_gpus, second_model_gpus


def _process_generations_text_to_text(
    generations: list[str], pattern: str, nebius_client, **kwargs
) -> list[str]:
    model_answers = []
    for gen in generations:
        if pattern is None or gen is None:
            logger.error(f"Invalid args:\nPattern:\t{pattern}\nResult:\t{gen}")
        if ending := re.search(pattern, gen):
            match_string = ending.group()
            index_answer_begin = match_string.index("(")
            answer = match_string[index_answer_begin : index_answer_begin + 3]
        else:
            answer = _extract_answer(gen, nebius_client)
        model_answers.append(answer)
    if "The answer is" in pattern:
        try:
            model_answers = [x[1] if x is not None else "-" for x in model_answers]
        except Exception as e:
            print(e)
    return model_answers


def _process_generations_image_to_text(
    generations: list[str], nebius_client, **kwargs
) -> list[str]:
    model_answers = []
    for gen in generations:
        if len(gen) == 1:
            model_answers.append(gen)
        else:
            answer = _extract_answer(gen, nebius_client)
            model_answers.append(answer)
    return model_answers


def _process_generations(
    task: Literal["text-to-text", "image-to-text"], **kwargs
) -> list[str]:
    PROCESS_GENERATIONS_MAP = {
        "text-to-text": _process_generations_text_to_text,
        "image-to-text": _process_generations_image_to_text,
    }
    if not task in PROCESS_GENERATIONS_MAP:
        raise ValueError("Unknown task: " + task)
    process_func = PROCESS_GENERATIONS_MAP[task]
    return process_func(**kwargs)


def _evaluate(
    dataset: Dataset, task, generations, pattern, nebius_client
) -> np.ndarray:
    model_answers = _process_generations(
        task=task, generations=generations, pattern=pattern, nebius_client=nebius_client
    )
    true_answers = dataset["output"]
    assert len(true_answers) == len(
        model_answers
    ), f"Lengths don't coincide! {len(true_answers)} for true answers and {len(model_answers)} for model predictions."
    scores = (np.array(true_answers) == model_answers).astype(int)
    return scores


def log_scores(dataset_name, model_name, scores):
    mean_scores = scores.mean()
    std = pow(mean_scores * (1 - mean_scores) / len(scores), 0.5)
    log = f"Dataset: {dataset_name}\tModel: {model_name}\tEvaluation score (accuracy): {mean_scores:.3f}\tstd: {std:.3f}"
    logger.info(log)


def evaluate(dataset_name, task, dataset, generations, nebius_client):
    if dataset_name in (
        "Aktsvigun/nebius_eval_mmlu_pro",
        "Aktsvigun/nebius_mmlu_pro_full",
    ):
        pattern = r"The answer is \(([A-Z])\)\.$"
    else:
        pattern = r"\([A-Z]\)$"
    return _evaluate(
        dataset=dataset,
        task=task,
        generations=generations,
        pattern=pattern,
        nebius_client=nebius_client,
    )


def run_single_evaluation(
    port: int,
    model_path: str,
    task: Literal["text-to-text", "image-to-text"],
    datasets: str,
    num_concurrent: int,
    max_tokens: int,
) -> dict[str, np.array]:
    logger.info(
        f"Port: {port}, Model path: {model_path}, Datasets: {datasets}, Num concurrent: {num_concurrent}, Max tokens: {max_tokens}"
    )
    client = Client(base_url=f"http://0.0.0.0:{port}/v1", api_key="kek")
    nebius_client = Client(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)
    all_results = {}

    for dataset_name in datasets:
        # Map dataset name if its shortened version is used
        dataset_name = DATASET_MAP.get(dataset_name, dataset_name)
        dataset = load_dataset(dataset_name, cache_dir=CACHE_DIR)
        if "train" in dataset.keys():
            dataset = dataset["train"]
        else:
            dataset = concatenate_datasets(list(dataset.values()))

        start_time = time()
        logger.info(
            f"Generating predictions on {dataset_name} using model {model_path} (port {port})..."
        )
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = {
                executor.submit(
                    process_example, task, messages, client, max_tokens
                ): idx
                for idx, messages in enumerate(dataset["input"])
            }
            generations = [None] * len(dataset["input"])

            with tqdm(
                total=len(futures),
                desc=f"Generating predictions using the model {model_path} (port {port})",
            ) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    generations[idx] = future.result()
                    pbar.update(1)

        logger.info(
            f"Generation on {dataset_name} using the model {model_path} (port {port}) took {time() - start_time:.1f} seconds."
        )

        with open(
            f'{RESULTS_SAVE_DIR}/{(model_path + "__" + dataset_name + f"__port{port}").replace("/", "__")}.json',
            "w",
        ) as f:
            json.dump(generations, f)

        scores = evaluate(
            dataset_name=dataset_name,
            task=task,
            dataset=dataset,
            generations=generations,
            nebius_client=nebius_client,
        )
        all_results[dataset_name] = scores
        log_scores(
            dataset_name=dataset_name,
            model_name="/".join(model_path.split("/")[-2:]),
            scores=scores,
        )

    return all_results


def run_parallel_evaluation(
    ports: list[int] | tuple[int],
    model_path: str,
    quant_model_path: str,
    task: Literal["text-to-text", "image-to-text"],
    num_concurrent: int,
    max_tokens: int,
    server2_exists: bool = False,
):
    """
    Returns:
        True, if quantized model doesn't have a quality drop or if only one model is provided
        False, if the quality of the quantized model is significantly lower
    """
    if not os.path.exists(RESULTS_SAVE_DIR):
        os.mkdir(RESULTS_SAVE_DIR)
    datasets = TASK_DATASETS_MAP[task]
    constant_kwargs = (datasets, num_concurrent, max_tokens)

    def run_eval(args):
        return run_single_evaluation(*args)

    if server2_exists:
        with ThreadPoolExecutor(max_workers=len(ports)) as executor:
            futures = [
                executor.submit(
                    run_eval, (ports[0], model_path, task) + constant_kwargs
                ),
                executor.submit(
                    run_eval, (ports[1], quant_model_path, task) + constant_kwargs
                ),
            ]
            results = [future.result() for future in futures]
    else:
        results = [
            run_single_evaluation(
                ports[0], model_path, task, datasets, num_concurrent, max_tokens
            )
        ]
    # Combine and average results from both servers
    combined_results = {port: res for port, res in zip(ports, results)}

    # Print final averaged results and add them to the tables
    df_quality_path = os.path.join(RESULTS_SAVE_DIR, "quality.csv")
    if os.path.exists(df_quality_path):
        df_quality = pd.read_csv(df_quality_path)
    else:
        df_quality = pd.DataFrame(columns=["model", "dataset", "score", "std"])
    models = [model_path, quant_model_path]
    for i, (port, port_results) in enumerate(combined_results.items()):
        for dataset_name, scores in port_results.items():
            mean_scores = np.mean(scores)
            std = pow(mean_scores * (1 - mean_scores) / len(scores), 0.5)
            df_quality.loc[len(df_quality)] = [
                models[i],
                dataset_name,
                mean_scores,
                std,
            ]
    df_quality.to_csv(df_quality_path, index=False)

    # Perform paired t-test if comparing the quantized model to the original one
    if server2_exists:
        df_quant_path = os.path.join(RESULTS_SAVE_DIR, "quantization_results.csv")
        if os.path.exists(df_quant_path):
            df_quant = pd.read_csv(df_quant_path)
        else:
            df_quant = pd.DataFrame(
                columns=[
                    "orig_model",
                    "quant_model",
                    "p_value",
                    "diff_is_significant",
                    "quality_orig",
                    "quality_quant",
                ]
            )

        p_values_lower_conf_level = []
        dataset_names = combined_results[ports[0]].keys()
        logger.info(f"{'^' * 30} Paired t-test results {'^' * 30}")
        for dataset in dataset_names:
            results_orig_model = combined_results[ports[0]][dataset]
            results_quant_model = combined_results[ports[1]][dataset]
            logger.info(f"{'-' * 15} Dataset: {dataset} {'-' * 15}")
            t_stat, p_value = stats.ttest_rel(results_orig_model, results_quant_model)
            diff_is_significant = p_value < CONFIDENCE_LEVEL
            p_values_lower_conf_level.append(diff_is_significant)
            logger.info(f"p-value: {p_value:.5f}")
            # Interpretation
            if diff_is_significant:
                logger.info(
                    f"The difference is statistically significant (p < {CONFIDENCE_LEVEL:.2f}). The quantized model likely has lower performance."
                )
            else:
                logger.info(
                    f"The difference is not statistically significant (p >= {CONFIDENCE_LEVEL:.2f}). The quantized model performance is similar."
                )
            df_quant.loc[len(df_quant)] = [
                model_path,
                quant_model_path,
                p_value,
                diff_is_significant,
                np.mean(results_orig_model),
                np.mean(results_quant_model),
            ]
        df_quant.to_csv(df_quant_path, index=False)
        if any(p_values_lower_conf_level):
            return False
    return True


def get_model_type(model_type: str):
    if model_type in ("CausalLM", "AutoModelForCausalLM"):
        return AutoModelForCausalLM
    else:
        module = importlib.import_module("transformers")
        return getattr(module, model_type)


def quantize_and_save(config: dict[str, str | int]):
    model_id = config["model_id"]
    model_type = config.get("model_type", "AutoModelForCausalLM")
    task = config["task"]
    save_dir = os.path.join(MODEL_SAVE_DIR, model_id)
    save_dir_fp8 = os.path.join(
        MODEL_SAVE_DIR, "nebius", model_id.split("/")[1] + "-FP8-Dynamic"
    )

    # Load model
    model_class = get_model_type(model_type)
    model = model_class.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", cache_dir=CACHE_DIR
    )
    if task == "text-to-text":
        tokenizer_or_processor = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR
        )
    elif task == "image-to-text":
        tokenizer_or_processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=CACHE_DIR
        )
    else:
        raise ValueError("Unknown task: " + task)
    if not os.path.exists(save_dir):
        logger.info("Saving the original model...")
        model.save_pretrained(save_dir)
        tokenizer_or_processor.save_pretrained(save_dir)
        logger.info("Successfully saved the original model. Starting quantization...")

    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per channel via ptq
    #   * quantize the activations to fp8 with dynamic per token
    ignore_modules = config["ignore_modules"]
    if not ignore_modules and task == "text-to-text":
        ignore_modules = ["lm_head"]
    elif not ignore_modules and task == "image-to-text":
        ignore_modules = ["lm_head", "re:visual.*"]
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore_modules
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    logger.info("Saving the quantized model...")
    # Save to disk in compressed-tensors format.
    model.save_pretrained(save_dir_fp8)
    tokenizer_or_processor.save_pretrained(save_dir_fp8)
    logger.info("Successfully saved.")
    del model
    gc.collect()
    cuda.empty_cache()


def main(config: dict[str, Any]):
    ### Validate the config is valid
    # Get available GPUs
    first_model_gpus, second_model_gpus = _assign_gpus(config["num_gpus_first_model"])
    ports = config["ports"].split(",")
    port1, port2 = ports
    if len(ports) != 2:
        raise ValueError(
            "Please provide 2 ports to run the quantized and non-quantized models."
        )

    # Quantize and save the model
    if config.get("do_quantize", DO_QUANTIZE):
        quantize_and_save(config)

    model_path = os.path.join(MODEL_SAVE_DIR, config["model_id"])
    quant_model_path = os.path.join(
        MODEL_SAVE_DIR, "nebius", config["model_id"].split("/")[1] + "-FP8-Dynamic"
    )
    max_tokens = config["max_tokens"]
    num_concurrent = config["num_concurrent"]
    task = config["task"]

    server_task_specific_args = SERVER_TASK_SPECIFIC_ARGS[task]
    # First server (always launches)
    logger.info(
        f"Serving model {model_path} on port {port1} and GPU devices {first_model_gpus}..."
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(first_model_gpus)
    server1_command = [
        "vllm",
        "serve",
        model_path,
        "--served_model_name",
        "model",
        "--port",
        port1,
        "-tp",
        str(len(first_model_gpus)),
        "--disable-log-requests",
        "--max-model-len",
        "32000",
        "--gpu-memory-utilization",
        "0.9",
    ] + server_task_specific_args

    server1 = subprocess.Popen(
        server1_command,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    
    # Give the server some time to initialize before checking health
    logger.info("Waiting 30 seconds for server with the original model to initialize...")
    sleep(30)
    
    # Second server (launches if sufficient number of GPUs is available)
    server2 = None
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(second_model_gpus)
    logger.info(
        f"Serving model {quant_model_path} on port {port2} and GPU devices {second_model_gpus}..."
    )
    if len(second_model_gpus) > 0:
        server2_command = [
            "vllm",
            "serve",
            quant_model_path,
            "--served_model_name",
            "model",
            "--port",
            port2,
            "-tp",
            str(len(second_model_gpus)),
            "--disable-log-requests",
            "--max-model-len",
            "32000",
            "--gpu-memory-utilization",
            "0.9",
        ] + server_task_specific_args

        server2 = subprocess.Popen(
            server2_command,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        
        # Give the server some time to initialize before checking health
        logger.info("Waiting 30 seconds for server with the quantized model to initialize...")
        sleep(30)

    # Ensure clean shutdown
    atexit.register(lambda: [p.terminate() for p in [server1, server2] if p])

    if not wait_server(port1):
        raise RuntimeError(f"Failed to start server 1 on port {port1}.")
    if server1.returncode is not None:
        raise RuntimeError(f"Failed to start server 1 on port {port1}.")
    if server2 is not None:
        if not wait_server(port2):
            raise RuntimeError(f"Failed to start server 2 on port {port2}.")
        if server2.returncode is not None:
            raise RuntimeError(f"Failed to start server 2 on port {port2}.")

    logger.info("Servers successfully started.")
    return run_parallel_evaluation(
        ports=ports,
        model_path=model_path,
        quant_model_path=quant_model_path,
        task=task,
        num_concurrent=num_concurrent,
        max_tokens=max_tokens,
        server2_exists=(server2 is not None),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", default="config.json", type=str, help="Path to config file"
    )
    parser.add_argument(
        "--model-id",
        default=None,
        type=str,
        help="Model ID on HuggingFace, e.g. `Qwen/Qwen2.5-VL-7B-Instruct`",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        type=str,
        help="Class from HF's `transformers` library used to load the model. Generally is provided in the model's card on HuggingFace in the \"Use with transformers\" section. For example, `Qwen2_5_VLForConditionalGeneration`",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help='Task for which the model used. Currently, only "text-to-text" and "image-to-text" values are supported.',
    )
    parser.add_argument(
        "--ports",
        default=None,
        type=int,
        help="Comma separated ports that will be used to serve the original & the quantized models, e.g. `8001,8002`",
    )
    parser.add_argument(
        "--num-gpus-first-model",
        default=None,
        type=int,
        help="Number of GPUs to use for the original model. Useful when limited in GPUs since the quantized model may occupy 2x times less GPUs. `-1` or empty value means automatic split of GPUs.",
    )
    parser.add_argument(
        "--num-concurrent",
        default=None,
        type=int,
        help="Number of concurrent request to send to each server.",
    )
    parser.add_argument(
        "--max-tokens",
        default=None,
        type=int,
        help="Maximum number of tokens to generate for all tasks.",
    )
    parser.add_argument(
        "--do-quantize",
        type=lambda x: str(x).lower() in ("true", "1", "t", "yes"),
        default=None,
        help="Whether to run quantization & model saving. Can be set using DO_QUANTIZE environment variable.",
    )
    parser.add_argument(
        "--ignore-modules",
        nargs="+",
        type=str,
        default=None,
        help='List of modules to ignore when quantizing the model. E.g. for most text-to-text models it is `["lm_head"]`, for Qwen-2.5-VL -- `["re:.*lm_head", "re:visual.*"]`. Should be taken from a relevant example from the `llm-compressor` repository. If you are in despair, ask @aktsvigun.',
    )

    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    for key in config:
        if (args_key := getattr(args, key)) is not None:
            config[key] = args_key

    verdict = main(config=config)
    if verdict:
        logger.info("Quantized model doesn't have a quality drop")
    else:
        logger.error("Quantized model has a quality drop")
