import time
import os
import mlflow

# Suppress TensorFlow logs (MUST be before importing keras/tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import os
import json
import argparse
import sys
import inspect

from src.evaluation.metrics import calculate_ranking_metrics, calculate_top_k_overlap
from src.utils.data_utils import prepare_dataset, extract_training_split_from_filename, estimate_period

from src.models.deep.lstm import LSTMDetector 
from src.models.deep.autoencoder import AutoencoderDetector 
from src.models.deep.tcn import TCNDetector

from src.models.traditional.isolation_forest import IsolationForestDetector 
from src.models.traditional.z_score import ZScoreDetector 
from src.models.traditional.local_outlier_factor import LOFDetector
from src.models.traditional.discords import MatrixProfileDiscordDetector

MODEL_REGISTRY = {
    "IsolationForest": IsolationForestDetector,
    "ZScore": ZScoreDetector,
    "LOF": LOFDetector,
    "MatrixProfile": MatrixProfileDiscordDetector,
    "LSTM": LSTMDetector,
    "Autoencoder": AutoencoderDetector,
    "TCN": TCNDetector
}

def get_user_selection():
    """
    Interactively asks the user to select models.
    """
    print("\n--- Available Models ---")
    model_names = list(MODEL_REGISTRY.keys())
    
    print("0. All Models")
    print("1. Traditional Models (IsolationForest, ZScore, LOF, MatrixProfile)")
    print("2. Deep Learning Models (LSTM, Autoencoder, TCN)")
    
    for i, name in enumerate(model_names):
        print(f"{i+3}. {name}")
        
    choice = input("\nEnter choice (comma separated, e.g. '3,4'): ").strip()
    
    selected_classes = []
    
    try:
        selections = [s.strip() for s in choice.split(',')]
        for s in selections:
            if s == '0':
                return list(MODEL_REGISTRY.items())
            elif s == '1':
                selected_classes.extend([(k, v) for k, v in MODEL_REGISTRY.items() if k in ["IsolationForest", "ZScore", "LOF", "MatrixProfile"]])
            elif s == '2':
                selected_classes.extend([(k, v) for k, v in MODEL_REGISTRY.items() if k in ["LSTM", "Autoencoder", "TCN"]])
            else:
                idx = int(s) - 3
                if 0 <= idx < len(model_names):
                    name = model_names[idx]
                    selected_classes.append((name, MODEL_REGISTRY[name]))
                else:
                    print(f"Warning: Invalid selection '{s}' ignored.")
    except Exception as e:
        print(f"Error parsing selection: {e}")
        return []
        
    return list(set(selected_classes))

def save_result(result, results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    n = 1
    while True:
        json_filename = os.path.join(results_dir, f"run_{n}.json")
        
        if not os.path.exists(json_filename):
            with open(json_filename, 'w') as f:
                json.dump([result], f, indent=4)
            print(f"Saved {result['Model']} to {json_filename}")
            break
        else:
            try:
                with open(json_filename, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
            
            model_exists = any(item.get("Model") == result["Model"] for item in existing_data)
            
            if model_exists:
                n += 1
                continue
            else:
                existing_data.append(result)
                with open(json_filename, 'w') as f:
                    json.dump(existing_data, f, indent=4)
                print(f"Appended {result['Model']} to {json_filename}")
                break
                
    return os.path.abspath(json_filename)

def run_benchmark(models_to_run, num_datasets=250, progress_callback=None, exclude_heavy=False):
    """
    Modular function to run the benchmark.
    """
    dataset_folder = "data"
    if not os.path.isdir(dataset_folder):
        print(f"Error: Dataset folder not found at {os.path.abspath(dataset_folder)}")
        return

    try:
        dataset_files_all_names = [f for f in os.listdir(dataset_folder) if f.endswith(".txt")]
        if not dataset_files_all_names:
            print(f"No .txt files found in {os.path.abspath(dataset_folder)}. Exiting.")
            return

        files_with_sizes = []
        for fname in dataset_files_all_names:
            fpath = os.path.join(dataset_folder, fname)
            try:
                files_with_sizes.append((fname, os.path.getsize(fpath)))
            except FileNotFoundError:
                continue
        
        files_with_sizes.sort(key=lambda x: x[1])
        
        if progress_callback:
            progress_callback(f"Running on {num_datasets} datasets.")
        else:
             print(f"Running on {num_datasets} datasets.")

        dataset_files = [fname for fname, size in files_with_sizes[:num_datasets]]

    except FileNotFoundError:
        print(f"Error: Dataset folder not found: {os.path.abspath(dataset_folder)}")
        return
    
    total_steps = len(models_to_run) * len(dataset_files)
    current_step = 0

    # --- MLFLOW SETUP ---
    mlflow.set_experiment("TSAD Benchmark")
    
    for model_name, ModelClass in models_to_run:
        with mlflow.start_run(run_name=f"{model_name}_Run"): 
            
            # Log Parameters (Configuration)
            mlflow.log_param("model_class", model_name)
            mlflow.log_param("dataset_count", len(dataset_files))
            mlflow.log_param("exclude_heavy", exclude_heavy)

            if progress_callback:
                progress_callback(f"Processing Model: {model_name}")
            else:
                print(f"\nProcessing Model: {model_name}")
            
            dataset_metrics_agg = {
                'sum_auc_roc': 0.0, 'sum_pr_auc': 0.0, 'sum_top_k': 0.0,
                'total_train_time': 0.0, 'total_pred_time': 0.0,
                'count': 0
            }

            for i, dataset_filename in enumerate(dataset_files):
                current_step += 1
                if progress_callback:
                    progress_callback(f"[{model_name}] Dataset {i+1}/{len(dataset_files)}")
                else:
                    print(f"  Dataset {i+1}/{len(dataset_files)}: {dataset_filename}")
                
                dataset_filepath = os.path.join(dataset_folder, dataset_filename)

                try:
                    all_data_df = prepare_dataset(dataset_filepath)
                    training_split_idx = extract_training_split_from_filename(dataset_filename)
                    train_df = all_data_df.iloc[:training_split_idx]
                    test_df = all_data_df.iloc[training_split_idx:]
                    
                    if test_df.empty:
                        continue

                    train_values = train_df['Value'].values
                    test_values = test_df['Value'].values
                    true_test_labels = test_df['is_anomaly'].values
                    
                    if len(train_values) > 10:
                        est_period = estimate_period(train_values)
                    else:
                        est_period = 50 
                    
                    window_size = int(max(32, min(est_period, 512, len(train_values)//2)))
                    
                    init_signature = inspect.signature(ModelClass.__init__)
                    init_params = init_signature.parameters
                    
                    kwargs = {}
                    if 'window_size' in init_params:
                        kwargs['window_size'] = window_size
                    if 'verbose' in init_params:
                        kwargs['verbose'] = 0
                    
                    try:
                        model_instance = ModelClass(**kwargs)
                    except Exception as e:
                        print(f"    Init failed with kwargs {kwargs}: {e}. Retrying without args.")
                        model_instance = ModelClass()
                    
                    start_train = time.time()
                    model_instance.fit(train_values)
                    dataset_metrics_agg['total_train_time'] += (time.time() - start_train)

                    start_pred = time.time()
                    
                    if model_name == "MatrixProfile" and 'window_size' in inspect.signature(model_instance.score).parameters:
                        anomaly_scores = model_instance.score(test_values, window_size=window_size)
                    else:
                        anomaly_scores = model_instance.score(test_values)
                    
                    dataset_metrics_agg['total_pred_time'] += (time.time() - start_pred)
                    
                    anomaly_scores = np.nan_to_num(anomaly_scores)

                    if len(anomaly_scores) != len(test_values):
                        print("    Score length mismatch. Skipping.")
                        continue

                    ranking = calculate_ranking_metrics(true_test_labels, anomaly_scores)
                    dataset_metrics_agg['sum_auc_roc'] += ranking['auc_roc']
                    dataset_metrics_agg['sum_pr_auc'] += ranking['pr_auc']
                    
                    top_k_hit = calculate_top_k_overlap(anomaly_scores, true_test_labels, k=None)
                    dataset_metrics_agg['sum_top_k'] += top_k_hit

                    dataset_metrics_agg['count'] += 1

                    if dataset_metrics_agg['count'] > 0 and (dataset_metrics_agg['count'] % 20 == 0):
                        if not progress_callback:
                            print(f"    Stats ({model_name} - {dataset_metrics_agg['count']}): "
                                f"AUC={dataset_metrics_agg['sum_auc_roc']/dataset_metrics_agg['count']:.3f}, "
                                f"TopK={dataset_metrics_agg['sum_top_k']/dataset_metrics_agg['count']:.3f}")

                except Exception as e:
                    print(f"      Error on {dataset_filename}: {e}")
                    continue

            if dataset_metrics_agg['count'] == 0:
                print(f"No valid results for {model_name}")
                continue

            count = dataset_metrics_agg['count']
            avg_auc = dataset_metrics_agg['sum_auc_roc'] / count
            avg_pr = dataset_metrics_agg['sum_pr_auc'] / count
            avg_topk = dataset_metrics_agg['sum_top_k'] / count
            avg_train_time = dataset_metrics_agg['total_train_time'] / count
            avg_pred_time = dataset_metrics_agg['total_pred_time'] / count

            res = {
                "Model": model_name,
                "AUC-ROC": avg_auc,
                "PR-AUC": avg_pr,
                "Top-K Hit Rate": avg_topk,
                "Avg Train Time (s)": avg_train_time,
                "Avg Predict Time (s)": avg_pred_time,
                "Datasets Processed": count
            }
            
            save_path = save_result(res)
            if not progress_callback:
                print(f"  Final {model_name}: AUC={res['AUC-ROC']:.3f}, Top-K={res['Top-K Hit Rate']:.2f}")

            # --- MLFLOW LOG METRICS ---
            mlflow.log_metric("auc_roc", avg_auc)
            mlflow.log_metric("pr_auc", avg_pr)
            mlflow.log_metric("top_k_hit_rate", avg_topk)
            mlflow.log_metric("train_time_s", avg_train_time)
            mlflow.log_metric("predict_time_s", avg_pred_time)

            # --- LOG ARTIFACTS ---
            temp_res_file = f"temp_res_{model_name}.json"
            with open(temp_res_file, 'w') as f:
                json.dump(res, f, indent=4)
            
            mlflow.log_artifact(temp_res_file)
            if os.path.exists(temp_res_file):
                os.remove(temp_res_file)

def run_pipeline():
    parser = argparse.ArgumentParser(description="Run Anomaly Detection Benchmark")
    parser.add_argument("--models", type=str, help="Comma separated list of model names to run (e.g. LSTM,TCN)")
    parser.add_argument("--group", type=str, choices=["all", "trad", "deep"], help="Run a specific group of models")
    parser.add_argument("--exclude_heavy", action="store_true", help="Exclude the 50 largest datasets to speed up deep learning")
    args = parser.parse_args()

    models_to_run = []

    if args.models:
        names = args.models.split(',')
        for n in names:
            n = n.strip()
            if n in MODEL_REGISTRY:
                models_to_run.append((n, MODEL_REGISTRY[n]))
            else:
                print(f"Warning: Model '{n}' not found.")
    elif args.group:
        if args.group == "all":
            models_to_run = list(MODEL_REGISTRY.items())
        elif args.group == "trad":
            models_to_run = [(k, v) for k, v in MODEL_REGISTRY.items() if k in ["IsolationForest", "ZScore", "LOF", "MatrixProfile"]]
        elif args.group == "deep":
            models_to_run = [(k, v) for k, v in MODEL_REGISTRY.items() if k in ["LSTM", "Autoencoder", "TCN"]]
    else:
        models_to_run = get_user_selection()
    
    if not models_to_run:
        print("No models selected. Exiting.")
        return

    print(f"\nSelected Models: {[n for n, _ in models_to_run]}")
    
    num_to_run = 250
    has_dl = any(name in ["LSTM", "Autoencoder", "TCN"] for name, _ in models_to_run)
    if args.exclude_heavy or (has_dl and not args.models and not args.group): 
        num_to_run = 200
        print(f"Running on {num_to_run} smallest datasets (excluding largest) for efficiency.")
    
    run_benchmark(models_to_run, num_datasets=num_to_run, exclude_heavy=args.exclude_heavy)

if __name__ == "__main__":
    run_pipeline()
