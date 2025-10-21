import json
import os
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from corect.config import *

USE_QUANTIZATION_METHODS = {
    # "32_full": "f32",
    "16_half": "f16",
    # "16_bfloat16": "bf16",
    "8_float8m3": "f8m3",
    # "8_float8m2": "f8m2",
    # "8_percentile": "8p",
    "8_equal_distance": "8",  # "8e",
    # "4_percentile": "4p",
    "4_equal_distance": "4",  # "4e",
    # "2_percentile": "2p",
    "2_equal_distance": "2",  # "2e",
    # "1_binary_median": "1m",
    "1_binary_zero": "1",  # "1z",
}
DATASETS = {
    "arguana": ["arguana"],
    "climate-fever": ["climate-fever"],
    "cqadupstack-android": ["cqadupstack-android"],
    "cqadupstack-english": ["cqadupstack-english"],
    "cqadupstack-gaming": ["cqadupstack-gaming"],
    "cqadupstack-gis": ["cqadupstack-gis"],
    "cqadupstack-mathematica": ["cqadupstack-mathematica"],
    "cqadupstack-physics": ["cqadupstack-physics"],
    "cqadupstack-programmers": ["cqadupstack-programmers"],
    "cqadupstack-stats": ["cqadupstack-stats"],
    "cqadupstack-tex": ["cqadupstack-tex"],
    "cqadupstack-unix": ["cqadupstack-unix"],
    "cqadupstack-webmasters": ["cqadupstack-webmasters"],
    "cqadupstack-wordpress": ["cqadupstack-wordpress"],
    "dbpedia": ["dbpedia"],
    "fever": ["fever"],
    "fiqa": ["fiqa"],
    "hotpotqa": ["hotpotqa"],
    "nfcorpus": ["nfcorpus"],
    "nq": ["nq"],
    "quora": ["quora"],
    "scidocs": ["scidocs"],
    "scifact": ["scifact"],
    "touche2020": ["touche2020"],
    "trec-covid": ["trec-covid"],
}


def collect_data(model_name: str, document_length: str, corpus_size: int):
    results_path = os.path.join(
        SHARE_RESULTS_FOLDER,
        model_name,
        document_length,
    )

    heatmap_data = defaultdict(dict)
    for metric in METRICS:
        heatmap_data[metric] = defaultdict(dict)

    for dim_dir in sorted(os.listdir(results_path)):
        dim_path = os.path.join(results_path, dim_dir)
        if not os.path.isdir(dim_path) or not dim_dir.startswith("dim="):
            continue

        dim = int(dim_dir.split("=")[1])

        for q_dir in sorted(os.listdir(dim_path), key=lambda x: int(x.split("=")[-1].split("_")[0]), reverse=True):
            q_path = os.path.join(dim_path, q_dir)
            if not os.path.isdir(q_path) or not q_dir.startswith("q="):
                continue

            q_method = q_dir.split("=")[1]
            if q_method not in USE_QUANTIZATION_METHODS:
                continue
            q = USE_QUANTIZATION_METHODS[q_method]

            json_file = os.path.join(q_path, f"{corpus_size}.json")

            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                    for metric in METRICS:
                        m = data.get(metric)
                        if m is not None:
                            heatmap_data[metric][q][dim] = m * 100

    return heatmap_data


def plot_heatmap(
    data: dict, model_name: str, document_length: str, corpus_size: int, metric: str
):
    plot_path = os.path.join(
        RESOURCES_FOLDER,
        "heatmaps",
        model_name,
        metric,
        f"{document_length}_{corpus_size}.pdf",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    df = (
        pd.DataFrame(data).T
        .sort_index(axis=1, ascending=False)
    )

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 14},
        cmap="crest_r",
    )

    # Get the colorbar and make it larger
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f"{METRICS[metric]} in %", fontsize=10)

    plt.xlabel("Dimensionality", fontsize=14, labelpad=10)
    plt.ylabel("Number of Bits per Dimension", fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


def plot_aggregated_heatmap(model_name: str, arrays: dict, cqa_values: dict, cqa_scores: dict):
    # Skip if no data is available
    if not arrays and not cqa_values:
        click.echo("No data available for the heatmap.")
        return

    for metric in METRICS:
        save_path = os.path.join(
            RESOURCES_FOLDER, "heatmaps", model_name, metric, "aggregated.pdf"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for dim, scores in cqa_scores[metric].items():
            for q, score in scores.items():
                cqa_scores[metric][dim][q] = np.mean(cqa_scores[metric][dim][q])

        plot_heatmap(cqa_scores[metric], model_name, "cqadupstack", "cqadupstack", metric)
        cqa = np.mean(np.vstack(cqa_values[metric]), axis=0)
        arrays[metric].append(cqa)
        arrays_metric = np.vstack(arrays[metric])
        means = np.mean(arrays_metric, axis=0).round(2)
        std = np.std(arrays_metric, axis=0).round(2)
        means_annot = []
        std_annot = []

        for idx, value in enumerate(means):
            if idx == 0:
                means_annot.append("_")
                std_annot.append("")
                continue
            means_annot.append(value)
            std_annot.append(f"±{std[idx]}")

        # shape = (len(cqa_scores[metric].keys()), len(cqa_scores[metric]["f32"].keys()))
        shape = (len(cqa_scores[metric].keys()), len(cqa_scores[metric]["f16"].keys()))
        means_annot = np.array(means_annot).reshape(shape)
        std_annot = np.array(std_annot).reshape(shape)
        df = pd.DataFrame(
            means.reshape(shape),
            # columns=list(cqa_scores[metric]["f32"].keys()),
            columns=list(cqa_scores[metric]["f16"].keys()),
            index=list(cqa_scores[metric].keys()),
        )
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            df,
            annot=False,
            fmt="",
            cmap="crest_r",
        )

        # Get the colorbar and make it larger
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(f"{METRICS[metric]} in %", fontsize=10)

        sns.heatmap(
            df,
            annot=means_annot,
            annot_kws={"va": "bottom", "fontsize": 14},
            fmt="",
            cmap="crest_r",
            cbar=False,
        )
        sns.heatmap(
            df,
            annot=std_annot,
            annot_kws={"va": "top", "fontsize": 13},
            fmt="",
            cmap="crest_r",
            cbar=False,
        )
        plt.xlabel("Dimensionality", fontsize=14, labelpad=10)
        plt.ylabel("Number of Bits per Dimension", fontsize=14, labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
        plt.close()


@click.command()
@click.argument("model_name", type=click.Choice(os.listdir(SHARE_RESULTS_FOLDER)))
def heatmap(model_name: str):
    """
    Visualize the relevance composition of the model's results.
    """
    arrays = {}
    cqa_values = {}
    cqa_scores = {}

    for metric in METRICS:
        arrays[metric] = []
        cqa_values[metric] = []
        cqa_scores[metric] = {}

    for document_length, corpus_sizes in DATASETS.items():

        for corpus_size in corpus_sizes:

            # Collect data from the model directory
            data = collect_data(model_name, document_length, corpus_size)

            for metric in METRICS:
                # Plot the heatmap chart
                plot_heatmap(data[metric], model_name, document_length, corpus_size, metric)

                if isinstance(corpus_size, int):
                    continue

                # full_prec_value = data[metric]["f32"][max(data[metric]["f32"].keys())]
                full_prec_value = data[metric]["f16"][max(data[metric]["f16"].keys())]
                data_metric = {
                    key: dict(sorted(value.items(), key=lambda x: x[0], reverse=True))
                    for key, value in data[metric].items()
                }
                # data_metric = dict(sorted(data_metric.items(), key=lambda x: x[0], reverse=True))
                values = []
                for dim, scores in data_metric.items():
                    for q, score in scores.items():
                        values.append(score - full_prec_value)
                if "cqadupstack" in corpus_size:
                    cqa_values[metric].append(np.array(values))
                    for dim, scores in data_metric.items():
                        if dim not in cqa_scores[metric].keys():
                            cqa_scores[metric][dim] = {}
                        for q, score in scores.items():
                            if q in cqa_scores[metric][dim].keys():
                                cqa_scores[metric][dim][q].append(score)
                            else:
                                cqa_scores[metric][dim][q] = [score]
                else:
                    arrays[metric].append(np.array(values))

    plot_aggregated_heatmap(model_name, arrays, cqa_values, cqa_scores)

    click.echo("Done")
