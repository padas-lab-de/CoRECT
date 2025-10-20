import json
import os
from collections import defaultdict

import click
import matplotlib.text as mtext
import numpy as np
from matplotlib import pyplot as plt

from corect.config import METRICS, SHARE_RESULTS_FOLDER, RESOURCES_FOLDER


USE_QUANTIZATION_METHODS = {
    "16_half": [(256, 4), (128, 8), (64, 16), (32, 32)],
    "16_bfloat16": [(256, 4), (128, 8), (64, 16), (32, 32)],
    "8_percentile": [(512, 4), (256, 8), (128, 16), (64, 32), (32, 64)],
    "8_equal_distance": [(512, 4), (256, 8), (128, 16), (64, 32), (32, 64)],
    "4_percentile": [(1024, 4), (512, 8), (256, 16), (128, 32), (64, 64)],
    "4_equal_distance": [(1024, 4), (512, 8), (256, 16), (128, 32), (64, 64)],
    "2_percentile": [(1024, 8), (512, 16), (256, 32), (128, 64)],
    "2_equal_distance": [(1024, 8), (512, 16), (256, 32), (128, 64)],
    "1_binary_median": [(1024, 16), (512, 32), (256, 64)],
    "1_binary_zero": [(1024, 16), (512, 32), (256, 64)],
    "1024_lsh": [(1024, 16)],
    "2048_lsh": [(1024, 8)],
    "4096_lsh": [(1024, 4)],
    "256_pca": [(1024, 4)],
    "128_pca": [(1024, 8)],
    "64_pca": [(1024, 16)],
    "512_8_pq": [(1024, 4)],
    "1024_2_pq": [(1024, 8)],
    "512_2_pq": [(1024, 16)],
    "128_8_pq": [(1024, 16)],
    "128_2_pq": [(1024, 32)],
}
DISPLAY_NAMES_AND_COLORS = {
    "32_full": ("FP32", "gray"),
    "16_half": ("FP16", "gray"),
    "16_bfloat16": ("BF16", "gray"),
    "8_percentile": ("8 Perc. Bin.", "darkblue"),
    "8_equal_distance": ("8 Equidistant Bin.", "darkgreen"),
    "4_percentile": ("4 Perc. Bin", "cornflowerblue"),
    "4_equal_distance": ("4 Equidistant Bin.", "mediumseagreen"),
    "2_percentile": ("2 Perc. Bin.", "lightskyblue"),
    "2_equal_distance": ("2 Equidistant Bin.", "palegreen"),
    "1_binary_median": ("1 Median Thresh.", "cyan"),
    "1_binary_zero": ("1 Zero Thresh.", "lime"),
    "1024_lsh": ("LSH", "red"),
    "2048_lsh": ("LSH", "red"),
    "4096_lsh": ("LSH", "red"),
    "768_lsh": ("LSH", "red"),
    "1536_lsh": ("LSH", "red"),
    "3072_lsh": ("LSH", "red"),
    "6144_lsh": ("LSH", "red"),
    "512_8_pq": ("PQ", "darkmagenta"),
    "1024_2_pq": ("PQ", "darkmagenta"),
    "512_2_pq": ("PQ", "darkmagenta"),
    "128_8_pq": ("PQ", "darkmagenta"),
    "128_2_pq": ("PQ", "darkmagenta"),
    "384_8_pq": ("PQ", "darkmagenta"),
    "768_8_pq": ("PQ", "darkmagenta"),
    "768_2_pq": ("PQ", "darkmagenta"),
    "384_2_pq": ("PQ", "darkmagenta"),
    "96_8_pq": ("PQ", "darkmagenta"),
    "96_2_pq": ("PQ", "darkmagenta"),
    "256_pca": ("PCA", "darkorange"),
    "128_pca": ("PCA", "darkorange"),
    "64_pca": ("PCA", "darkorange"),
    "192_pca": ("PCA", "darkorange"),
    "96_pca": ("PCA", "darkorange"),
    "48_pca": ("PCA", "darkorange"),
}
X_POS = {
    "8_percentile": -0.3,
    "8_equal_distance": 0.3,
    "4_percentile": 0.5,
    "4_equal_distance": -0.5,
    "2_percentile": -0.2,
    "2_equal_distance": 0.2,
    "1_binary_median": -0.4,
    "1_binary_zero": 0.4,
}
GROUPS = {
    "PQ": {"marker": "o", "name": "pq"},
    "LSH": {"marker": "s", "name": "lsh"},
    "PCA": {"marker": "*", "name": "pca"},
    "FP": {"marker": "H", "name": "16_"},
    "EDB": {"marker": "D", "name": "equal"},
    "PB": {"marker": "v", "name": "percentile"},
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


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title


def collect_data(model_name: str, document_length: str, corpus_size: int):
    results_path = os.path.join(
        SHARE_RESULTS_FOLDER,
        model_name,
        document_length,
    )
    data = defaultdict(dict)

    for metric in METRICS:
        data[metric] = defaultdict(dict)

    for q_method in USE_QUANTIZATION_METHODS:
        for dim, compression_ratio in USE_QUANTIZATION_METHODS[q_method]:
            for dim_dir in os.listdir(results_path):
                dim_path = os.path.join(results_path, dim_dir)
                if not os.path.isdir(dim_path) or not dim_dir.startswith("dim="):
                    continue

                dir_dim = int(dim_dir.split("=")[1])
                q_dir = f"q={q_method}"

                if dim != dir_dim or not q_dir in os.listdir(dim_path):
                    continue

                q_path = os.path.join(dim_path, q_dir)
                json_file = os.path.join(q_path, f"{corpus_size}.json")

                if os.path.isfile(json_file):
                    with open(json_file, "r") as f:
                        json_data = json.load(f)
                        for metric in METRICS:
                            m = json_data.get(metric)
                            if m is not None:
                                data[metric][compression_ratio][q_method] = m

    return data


def _merge_values(values: dict, calculate_mean: bool):
    aggregated = {}
    for metric, data_list in values.items():
        aggregated[metric] = defaultdict(dict)
        for data in data_list:
            for compression_ratio, q_methods in data.items():
                for q_method in q_methods:
                    if compression_ratio in aggregated[metric] and q_method in aggregated[metric][compression_ratio]:
                        aggregated[metric][compression_ratio][q_method].append(data[compression_ratio][q_method])
                    else:
                        aggregated[metric][compression_ratio][q_method] = [data[compression_ratio][q_method]]
    if calculate_mean:
        for metric, compression_ratios in aggregated.items():
            for compression_ratio, q_methods in compression_ratios.items():
                for q_method in q_methods:
                    aggregated[metric][compression_ratio][q_method] = np.mean(
                        aggregated[metric][compression_ratio][q_method]
                    )
    return aggregated


def _generate_plot(data: dict, model_name: str):
    if not data:
        click.echo("No data to plot!")
        return

    for metric in METRICS:
        save_path = os.path.join(
            RESOURCES_FOLDER, "pareto", model_name, metric, "aggregated.pdf"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 6))

        for i, (group_name, group_info) in enumerate(GROUPS.items()):
            marker = group_info["marker"]
            name = group_info["name"]

            for compression_ratio, q_methods in data[metric].items():
                bound = compression_ratio / 4
                for j, q_method in enumerate(q_methods):
                    if name in q_method or (name == "percentile" and q_method == "1_binary_median") or (
                            name == "equal" and q_method == "1_binary_zero"):
                        color = DISPLAY_NAMES_AND_COLORS[q_method][1]
                        x = compression_ratio
                        if group_name in ["EDB", "PB"]:
                            x += X_POS[q_method] * bound
                            if len(q_methods) > 10:
                                x += 0.05 * np.sign(X_POS[q_method]) * bound
                        if "jina" in model_name and compression_ratio == 4 and group_name == "FP":
                            x -= 0.2 * bound
                        elif "e5" in model_name and compression_ratio == 16 and group_name == "FP":
                            x -= 0.2 * bound
                        y = np.mean(data[metric][compression_ratio][q_method])
                        plt.scatter(x, y, label=DISPLAY_NAMES_AND_COLORS[q_method][0], marker=marker, color=color,
                                    edgecolor='black', s=100, alpha=0.7)
                plt.axvspan(compression_ratio - 0.7 * bound, compression_ratio + 0.7 * bound, color='whitesmoke',
                            alpha=0.1, lw=0, zorder=-1)

        plt.xscale('log', base=2)
        plt.xlabel("Compression Ratio", fontsize=16)
        plt.ylabel("Recall@100", fontsize=16)
        xticks = list(data[metric].keys())
        plt.xticks(xticks, [str(xtick) for xtick in xticks], fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        labels = list(unique.keys())[0:3] + [""] + list(unique.keys())[3:]
        handles = list(unique.values())[0:3] + ["With Vector Truncation:"] + list(unique.values())[3:]
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11,
                   title="Compression Methods", title_fontsize=12, handler_map={str: LegendTitle({'fontsize': 11})})
        plt.tight_layout()
        plt.savefig(save_path)
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
        plt.close()


@click.command()
@click.argument("model_name", type=click.Choice(os.listdir(SHARE_RESULTS_FOLDER)))
def pareto_plot(model_name: str):
    arrays = {}
    cqa_values = {}

    for metric in METRICS:
        arrays[metric] = []
        cqa_values[metric] = []

    for document_length, corpus_sizes in DATASETS.items():

        for corpus_size in corpus_sizes:

            # Collect data from the model directory
            data = collect_data(model_name, document_length, corpus_size)

            for metric in data:
                if "cqadupstack" in corpus_size:
                    cqa_values[metric].append(data[metric])
                else:
                    arrays[metric].append(data[metric])
    cqa_scores = _merge_values(cqa_values, True)
    arrays = _merge_values(arrays, False)

    for metric, compression_ratios in arrays.items():
        for compression_ratio, q_methods in compression_ratios.items():
            for q_method in q_methods:
                arrays[metric][compression_ratio][q_method].append(cqa_scores[metric][compression_ratio][q_method])

    _generate_plot(arrays, model_name)
