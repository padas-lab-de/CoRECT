import json
import os
from collections import defaultdict
from typing import Dict, List

import click
import pandas as pd
from scipy.stats import ttest_rel

from corect.config import SHARE_RESULTS_FOLDER, METRICS, RESOURCES_FOLDER

COMPARISONS = [("8_percentile", "8_equal_distance"), ("4_percentile", "4_equal_distance"),
               ("2_percentile", "2_equal_distance"), ("1_binary_median", "1_binary_zero")]
DATASETS = {
    "passage": [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ],
    "document": [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ],
    "beir": [
        "arguana",
        "climate-fever",
        "cqadupstack-android",
        "cqadupstack-english",
        "cqadupstack-gaming",
        "cqadupstack-gis",
        "cqadupstack-mathematica",
        "cqadupstack-physics",
        "cqadupstack-programmers",
        "cqadupstack-stats",
        "cqadupstack-tex",
        "cqadupstack-unix",
        "cqadupstack-webmasters",
        "cqadupstack-wordpress",
        "dbpedia",
        "fever",
        "fiqa",
        "hotpotqa",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "touche2020",
        "trec-covid",
    ]
}


def _collect_pca(save_path: str, dataset_name: str, results_pca: dict):
    """
    Loads the performance results for PCA and the corresponding MRL dimensionality reduction, then appends them to the
    dictionary.

    Args:
        save_path: The path where the PCA results are stored.
        dataset_name: The name of the dataset for which results should be loaded.
        results_pca: The dictionary to which results should be added.
    """
    dir_dims = [int(x.split("=")[1]) for x in os.listdir(save_path)]
    pca_path = os.path.join(save_path, f"dim={max(dir_dims)}")

    for q_method in os.listdir(pca_path):
        if "pca" in q_method:
            dim = q_method.split("_")[0].replace("q=", "")
            pca_json = os.path.join(pca_path, q_method, f"{dataset_name}.json")
            full_prec_json = os.path.join(save_path, f"dim={dim}", "q=16_half", f"{dataset_name}.json")
            if os.path.isfile(full_prec_json) and os.path.isfile(pca_json):
                with open(full_prec_json, "r") as f:
                    data = json.load(f)
                    for metric in METRICS:
                        m = data.get(metric)
                        if m is not None:
                            results_pca[metric]["q=16_half"].append(m * 100)
                with open(pca_json, "r") as f:
                    data = json.load(f)
                    for metric in METRICS:
                        m = data.get(metric)
                        if m is not None:
                            results_pca[metric]["pca"].append(m * 100)


def _collect_comparisons(save_path: str, dataset_name: str, results: dict):
    """
    Loads the performance results for each tuple of methods that should be compared and stores them in the given
    dictionary.

    Args:
        save_path: The path where the results are stored.
        dataset_name: The name of the dataset for which results should be loaded.
        results: The dictionary to which results should be added.
    """
    for (method1, method2) in COMPARISONS:
        method = f"{method1}-{method2}"
        for dim_dir in os.listdir(save_path):
            json_file1 = os.path.join(save_path, dim_dir, f"q={method1}", f"{dataset_name}.json")
            json_file2 = os.path.join(save_path, dim_dir, f"q={method2}", f"{dataset_name}.json")
            if os.path.isfile(json_file1) and os.path.isfile(json_file2):
                metrics1 = {}
                with open(json_file1, "r") as f:
                    data = json.load(f)
                    for metric in METRICS:
                        metrics1[metric] = data.get(metric) * 100
                with open(json_file2, "r") as f:
                    data = json.load(f)
                    for metric in METRICS:
                        m = data.get(metric) * 100
                        if method in results[metric]:
                            results[metric][method][0].append(metrics1[metric])
                            results[metric][method][1].append(m)
                        else:
                            results[metric][method] = ([], [])
                            results[metric][method][0].append(metrics1[metric])
                            results[metric][method][1].append(m)


def _initialize_results_dict(pca: bool) -> Dict[str, Dict[str, List[float]]]:
    """
    Initializes the results dictionary with an entry per metric.

    Args:
        pca: Whether the dictionary should contain results for PCA or for comparing other methods.

    Returns:
        The empty results dictionary.
    """
    results = {}
    for metric in METRICS:
        if pca:
            results[metric] = defaultdict(list)
        else:
            results[metric] = defaultdict(tuple)
    return results


@click.command()
@click.argument("model_name", type=click.Choice(os.listdir(SHARE_RESULTS_FOLDER)))
def statistics(model_name: str):
    """
    Calculates a paired t-test between results of two different compression methods for the different datasets and
    saves the results in a CSV file.

    Args:
        model_name: The name of the model for which t-test results should be calculated.
    """
    for corpus in DATASETS:
        pca_values = []
        values = defaultdict(list)
        for dataset in DATASETS[corpus]:
            results = _initialize_results_dict(False)
            results_pca = _initialize_results_dict(True)
            dataset_name = dataset if corpus == "beir" else corpus
            save_path = os.path.join(SHARE_RESULTS_FOLDER, model_name, dataset_name)
            _collect_pca(save_path, dataset, results_pca)
            _collect_comparisons(save_path, dataset, results)
            for metric in METRICS:
                t_stat, p_value = ttest_rel(results_pca[metric]["pca"], results_pca[metric]["q=16_half"])
                significant = p_value < 0.05
                pca_values.append([metric, dataset, t_stat, p_value, significant])
                click.echo(f"T-test values for {METRICS[metric]} on {corpus}-{dataset}: {t_stat}, {p_value}.")
                for methods in results[metric]:
                    t_stat, p_value = ttest_rel(results[metric][methods][0], results[metric][methods][1])
                    significant = p_value < 0.05
                    values[methods].append([methods, metric, dataset, t_stat, p_value, significant])
                    click.echo(f"T-test values for {METRICS[metric]} on {corpus}-{dataset} with {methods}: {t_stat}, "
                               f"{p_value}.")
        os.makedirs(os.path.join(RESOURCES_FOLDER, "t-test", model_name), exist_ok=True)
        df = pd.DataFrame(pca_values, columns=["metric", "dataset", "t-test", "p-value", "significant"])
        df.to_csv(os.path.join(RESOURCES_FOLDER, "t-test", model_name, f"pca_{corpus}.csv"), index=False)
        for methods in values:
            df = pd.DataFrame(values[methods], columns=["method", "metric", "dataset", "t-test", "p-value",
                                                        "significant"])
            df.to_csv(os.path.join(RESOURCES_FOLDER, "t-test", model_name, f"{methods}_{corpus}.csv"), index=False)
