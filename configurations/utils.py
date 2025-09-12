import os
import json
import numbers
from typing import Dict

def load_config(file_path: str) -> dict:
    """
    Loads a JSON configuration file and returns it as a dictionary.

    Args:
        file_path (str): Full path to the JSON configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the file extension is not .json
    """
    if not file_path.endswith(".json"):
        raise ValueError(f"Expected a .json file, got: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

def runConfigHealthChecks(config: dict) -> bool:
    """
    Perform a health check on a YOLOv1 configuration dictionary.
    Ensures required keys exist and values have correct types.

    Args:
        config (dict): Configuration dictionary returned by load_config().

    Returns:
        bool: True if config passes all checks, False otherwise.
    """

    required_top_keys = ["accelerator", "model", "training", "dataset", "callbacks"]
    for key in required_top_keys:
        if key not in config:
            print(f"[ERROR] Missing top-level key: {key}")
            return False

    if config["accelerator"] not in ["gpu", "cpu"]:
        print("[ERROR] 'accelerator' should be 'gpu' or 'cpu'")
        return False

    return (
        runModelHealthChecks(config) and
        runTrainingHealthChecks(config) and
        runDatasetHealthChecks(config)
    )

def runModelHealthChecks(config: Dict):
    model_keys = ["input_w", "input_h", "backbone", "split_size", "num_boxes"]
    for key in model_keys:
        if key not in config["model"]:
            print(f"[ERROR] Missing model key: {key}")
            return False
    if not isinstance(config["model"]["input_w"], int) or not isinstance(config["model"]["input_h"], int):
        print("[ERROR] 'input_w' and 'input_h' must be integers")
        return False
    return True

def runTrainingHealthChecks(config: Dict):
    training_keys = ["precision", "epochs", "batch_size", "validation_epochs", "hyperparams"]
    for key in training_keys:
        if key not in config["training"]:
            print(f"[ERROR] Missing training key: {key}")
            return False

    return (
        runTrainingHyperparamsHealthChecks(config) and
        runTrainingOptimHealthChecks(config)
    )

def runTrainingHyperparamsHealthChecks(config: Dict):
    hyperparams = config["training"]["hyperparams"]
    required_hyper_keys = ["lambda_coord", "lambda_noobj", "optim", "lr_scheduler"]
    for key in required_hyper_keys:
        if key not in hyperparams:
            print(f"[ERROR] Missing hyperparam key: {key}")
            return False

    return (
        runTrainingOptimHealthChecks(config) and
        runTrainingLrSchedulerHealthChecks(config)
    )

def runTrainingOptimHealthChecks(config: Dict):
    available_optims = ['adam','adamw']
    hyperparams = config["training"]["hyperparams"]
    optim_keys = ["name", "weight_decay", "learning_rate", "adam_beta1", "adam_beta2"]
    for key in optim_keys:
        if key not in hyperparams["optim"]:
            print(f"[ERROR] Missing optimizer key: {key}")
            return False

        if key == "name" and hyperparams["optim"][key] not in available_optims:
            print(f"[ERROR] Optim key {key} must be a value from {available_optims}. It was {hyperparams['optim'][key]}.")
            return False

        if key in ["weight_decay", "learning_rate", "adam_beta1", "adam_beta2"]:
            if not isinstance(hyperparams["optim"][key], numbers.Number):
                print(f"[ERROR] Optim key {key} must be a number")
                return False

    return True

def runTrainingLrSchedulerHealthChecks(config: Dict):
    lr_sched = config["training"]["hyperparams"]["lr_scheduler"]

    if "name" not in lr_sched or "params" not in lr_sched:
        print("[ERROR] lr_scheduler must have 'name' and 'params' keys.")
        return False

    name = lr_sched["name"]
    if name == None:
        print("[WARNING] lr_scheduler is set to None → no scheduler will be used.")
        return True

    name = name.lower()
    params = lr_sched["params"]

    allowed_schedulers = ["cosineannealinglr", "steplr", None]
    if name not in allowed_schedulers:
        print(f"[ERROR] Unsupported lr_scheduler '{name}'. Allowed schedulers: {allowed_schedulers}")
        return False

    required_params = []
    print(f'LR SCHEDULER NAME: {name}')
    if name == "cosineannealinglr":
        required_params = ["eta_min", "last_epoch"]
    elif name == "steplr":
        required_params = ["step_size", "gamma", "last_epoch"]

    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        print(f"[ERROR] Missing parameters for {name}: {missing_params}")
        return False

    # Optional: check numeric types
    for p in required_params:
        if not isinstance(params[p], numbers.Number):
            print(f"[ERROR] Parameter '{p}' must be a number, got {type(params[p])}")
            return False

    return True

def runDatasetHealthChecks(config: Dict):
    dataset_keys = ["name", "path", "num_workers", "num_classes", "data_chs", "prepare_per_node"]
    for key in dataset_keys:
        if key not in config["dataset"]:
            print(f"[ERROR] Missing dataset key: {key}")
            return False
    return True

def runCallbacksHealthChecks(config: Dict):
    if "early_stopping" not in config["callbacks"] or "checkpoint" not in config["callbacks"]:
        print("[ERROR] Callbacks must include 'early_stopping' and 'checkpoint'")
        return False
    return True
