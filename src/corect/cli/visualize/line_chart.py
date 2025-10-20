import json
import os
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from corect.config import *


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
}
COMPRESSION_PAIRS_16 = [
    (1024, "16_bfloat16"),
    (1024, "1_binary_median"),
    (512, "2_percentile"),
    (256, "4_percentile"),
    (128, "8_percentile"),
    (1024, "32_pca"),
    (1024, "512_2_pq"),
    (1024, "128_8_pq"),
    (1024, "1024_lsh")
]
COMPRESSION_PAIRS_DIM = [
    (1024, "16_bfloat16"),
    (512, "16_bfloat16"),
    (256, "16_bfloat16"),
    (128, "16_bfloat16"),
    (64, "16_bfloat16"),
    (32, "16_bfloat16"),
]
COMPRESSION_PAIRS_Q = [
    (1024, "16_bfloat16"),
    (1024, "8_percentile"),
    (1024, "4_percentile"),
    (1024, "2_percentile"),
    (1024, "1_binary_median"),
]
DISPLAY_NAMES = {
    "32_full": "32",
    "16_half": "16",
    "16_bfloat16": "16",
    "8_percentile": "8",
    "4_percentile": "4",
    "2_percentile": "2",
    "1_binary_median": "1",
}
COMPUTE_METRICS = {
    "ndcg_at_10": lambda data: compute_ndcg(data, k=10),
    "recall_at_100": lambda data: compute_recall(data, k=100),
    "recall_at_1000": lambda data: compute_recall(data, k=1000),
}
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
THRESHOLD_P_VALUE = 0.05


def compute_ndcg(data, k=10):
    """
    Compute NDCG at k for the given data.
    """
    ndcg_values = []

    rankings = data["rankings"]
    for _, ranking in rankings.items():
        relevant = ranking["relevant"]
        set_ranks = set()
        for _, rank in relevant.items():
            set_ranks.add(rank)

        if not set_ranks:
            ndcg_values.append(0.0)
            continue

        dcg = sum(1 / np.log2(rank + 1) for rank in set_ranks if rank <= k)
        idcg = sum(1 / np.log2(i + 1) for i in range(1, min(k, 10) + 1))

        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    return ndcg_values


def compute_recall(data, k=100):
    """
    Compute recall at k for the given data.
    """
    recall_values = []

    rankings = data["rankings"]
    for _, ranking in rankings.items():
        relevant = ranking["relevant"]
        set_ranks = set()
        for _, rank in relevant.items():
            set_ranks.add(rank)

        recall_values.append(
            sum(1 for r in set_ranks if r <= k) / 10 if set_ranks else 0.0
        )

    return recall_values


def add_data(results: dict, corpus_sizes: list, dim: int, q: str, results_path: str, metric: str, key: str):
    dim_dir = f"dim={dim}"
    q_dir = f"q={q}"
    for corpus_size in corpus_sizes:
        file_path = os.path.join(
            results_path, dim_dir, q_dir, f"{corpus_size}.json"
        )

        with open(file_path, "r") as f:
            data = json.load(f)
            m = COMPUTE_METRICS[metric](data)
        results[key][(dim, q)][corpus_size] = m


def collect_data(
    model_name: str, document_length: str, corpus_sizes: list, metric: str
):
    results_path = os.path.join(
        SHARE_RESULTS_FOLDER,
        model_name,
        document_length,
    )

    results = {
        "16": defaultdict(dict),
        "dim": defaultdict(dict),
        "q": defaultdict(dict),
    }

    for dim, q in COMPRESSION_PAIRS_16:
        add_data(results, corpus_sizes, dim, q, results_path, metric, "16")

    for dim, q in COMPRESSION_PAIRS_DIM:
        add_data(results, corpus_sizes, dim, q, results_path, metric, "dim")

    for dim, q in COMPRESSION_PAIRS_Q:
        add_data(results, corpus_sizes, dim, q, results_path, metric, "q")

    return results


def plot_lines(
    data_dict: dict,
    model_name: str,
    document_length: str,
    corpus_sizes: list,
    metric: str,
):
    plot_path = os.path.join(
        RESOURCES_FOLDER,
        "line_charts",
        model_name,
        metric,
        document_length,
        "{}.pdf",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    x = corpus_sizes
    xticks = [f"{corpus_size:,}" for corpus_size in corpus_sizes]

    for data_index, data in data_dict.items():

        plt.figure(figsize=(6, 5))
        baseline = None
        for i, ((dim, q), results_dict) in enumerate(data.items()):
            results = [results_dict[corpus_size] for corpus_size in corpus_sizes]
            values = [np.mean(r) for r in results]

            if not baseline:
                baseline = results
                p_values = [1.0] * len(results)
            else:
                p_values = [
                    wilcoxon(b, r, alternative="greater")[1]  # One-sided test
                    for b, r in zip(baseline, results)
                ]

            plt.plot(x, values, marker="o", label=f"dim={dim}, q={q}", color=COLORS[i])

            # Add markers for significant differences
            x_x = [x[j] for j in range(len(x)) if p_values[j] < THRESHOLD_P_VALUE]
            x_y = [values[j] for j in range(len(values)) if p_values[j] < THRESHOLD_P_VALUE]
            plt.plot(x_x, x_y, marker="*", color=COLORS[i], linestyle=None, markersize=12)

        plt.xscale("log")
        plt.xticks(
            x,
            xticks,
            rotation=30,
            fontsize=14,
        )
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(plot_path.format(data_index))
        plt.savefig(plot_path.format(data_index).replace(".pdf", ".png"), dpi=300)
        plt.close()


def plot_combined_lines(model_name: str, metric: str):
    corpus_passage = DATASETS["passage"]
    corpus_document = DATASETS["document"]

    # Collect relevant data
    data_passage = collect_data(model_name, "passage", corpus_passage, metric)
    data_document = collect_data(model_name, "document", corpus_document, metric)

    # Create subplots with shared y-axis
    _, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 0.8]},
    )

    titles = [
        "(a) Passage Retrieval on Full Precision Embeddings",
        "(b) Passage Retrieval on Quantized Embeddings",
        "(c) Document Retrieval on Quantized Embeddings",
    ]
    legend_titles = [
        "Dimensionality",
        "Number of bits\nper dimension",
        "Number of bits\nper dimension",
    ]
    keys = ["dim", "q", "q"]
    data_sources = [data_passage, data_passage, data_document]
    corpora = [corpus_passage, corpus_passage, corpus_document]

    def get_label(index, dim, q):
        if index == 0:
            return dim
        else:
            return DISPLAY_NAMES[q]

    for i, ax in enumerate(axes):
        key = keys[i]
        data = data_sources[i]
        corpus_sizes = corpora[i]

        baseline = None
        for j, ((dim, q), results_dict) in enumerate(data[key].items()):
            results = [results_dict[corpus_size] for corpus_size in corpus_sizes]
            values = [np.mean(r) for r in results]

            if not baseline:
                baseline = results
                p_values = [1.0] * len(results)
            else:
                p_values = [
                    wilcoxon(b, r, alternative="greater")[1]  # One-sided test
                    for b, r in zip(baseline, results)
                ]

            if dim == 1024 and q == "16_bfloat16":
                ax.plot(
                    corpus_sizes,
                    values,
                    marker="o",
                    label=get_label(i, dim, q),
                    linewidth=4,
                    markersize=10,
                )
            else:
                ax.plot(corpus_sizes, values, marker="o", label=get_label(i, dim, q), color=COLORS[j])
                # Add markers for significant differences
                x_x = [corpus_sizes[j] for j in range(len(corpus_sizes)) if p_values[j] < THRESHOLD_P_VALUE]
                x_y = [values[j] for j in range(len(values)) if p_values[j] < THRESHOLD_P_VALUE]
                ax.plot(x_x, x_y, color=COLORS[j], marker="*", linestyle=None, markersize=12)

        ax.set_xscale("log")
        ax.set_xticks(corpus_sizes)
        ax.set_xticklabels([f"{x:,}" for x in corpus_sizes], rotation=30, fontsize=12)
        ax.set_xlabel("Corpus size", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(titles[i], fontsize=12.5, pad=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        if i == 0:
            ax.set_ylabel("Recall@100", fontsize=14)

        legend = ax.legend(title=legend_titles[i], loc="upper right", ncol=2, fontsize=8)
        legend.get_title().set_ha("center")
        legend.get_title().set_fontsize(10)

    plt.tight_layout()

    output_path = os.path.join(
        RESOURCES_FOLDER, "line_charts", model_name, metric, "combined.pdf"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


@click.command()
@click.argument("model_name", type=click.Choice(os.listdir(SHARE_RESULTS_FOLDER)))
def line_chart(model_name: str):
    """
    Visualize the relevance composition of the model's results.
    """
    for metric in METRICS:

        # Plot combined line charts
        plot_combined_lines(model_name, metric)

        for document_length, corpus_sizes in DATASETS.items():

            # Collect data from the model directory
            data = collect_data(model_name, document_length, corpus_sizes, metric)

            # Plot the line chart
            plot_lines(data, model_name, document_length, corpus_sizes, metric)

    click.echo("Done")
