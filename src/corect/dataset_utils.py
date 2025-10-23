from collections import defaultdict
from typing import Dict, Tuple, List

import click
from datasets import load_dataset

CoRE = {
    "passage": {
        "pass_core": 10_000,
        "pass_10k": 10_000,
        "pass_100k": 100_000,
        "pass_1M": 1_000_000,
        "pass_10M": 10_000_000,
        "pass_100M": 100_000_000,
    },
    "document": {
        "doc_core": 10_000,
        "doc_10k": 10_000,
        "doc_100k": 100_000,
        "doc_1M": 1_000_000,
        "doc_10M": 10_000_000,
    },
}

BEIR = [
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
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "touche2020",
    "trec-covid",
]

CoRE_NAME = "core"
BEIR_NAME = "beir"

DATASETS = {CoRE_NAME: CoRE, BEIR_NAME: BEIR}


def _load_beir_data(
    dataset_sub_corpus: str,
    split: str,
) -> Tuple[defaultdict, Dict[str, str], defaultdict, defaultdict]:
    """
    Loads the test or dev split of the given BEIR dataset from the MTEB huggingface repo and returns the corresponding
    corpus, queries and qrels.

    Args:
        dataset_sub_corpus: The name of the BEIR dataset to load.

    Returns:
        The loaded corpus, queries and qrels.
    """
    dataset_queries = load_dataset(f"mteb/{dataset_sub_corpus}", "queries")
    dataset_qrels = load_dataset(f"mteb/{dataset_sub_corpus}", "default")
    dataset_corpus = load_dataset(f"mteb/{dataset_sub_corpus}", "corpus")

    if split in dataset_queries:
        dataset_queries = dataset_queries[split]
    if split in dataset_qrels:
        dataset_qrels = dataset_qrels[split]
    if split in dataset_corpus:
        dataset_corpus = dataset_corpus[split]

    # Transform dataset
    qrels = defaultdict(dict)
    for q in dataset_qrels:
        query_id = q["query-id"]
        corpus_id = q["corpus-id"]
        qrels[query_id][corpus_id] = int(q["score"])

    queries = {q["_id"]: q["text"] for q in dataset_queries["queries"] if q["_id"] in qrels.keys()}

    corpora = defaultdict(dict)
    for d in dataset_corpus["corpus"]:
        corpora[dataset_sub_corpus][d["_id"]] = {
            "title": d["title"],
            "text": d["text"],
        }

    return corpora, queries, qrels, qrels


def _load_core_data(
    dataset_sub_corpus: str,
) -> Tuple[defaultdict, Dict[str, str], defaultdict, defaultdict]:
    """
    Loads the corpus, queries and qrels of the given CoRE dataset (passage or document) from the corresponding
    huggingface repo for all specified dataset sizes, i.e. 10k, 100k, 1M etc.

    Args:
        dataset_sub_corpus: The CoRE dataset to load, i.e., passage or document.

    Returns:
        The loaded corpus, queries, qrels and relevant qrels.
    """
    # Load queries dataset
    dataset_queries = load_dataset("PaDaS-Lab/CoRE", "queries")["test"]

    # Load the qrels dataset
    dataset_qrels = load_dataset("PaDaS-Lab/CoRE", "qrels")[dataset_sub_corpus]

    # Transform the datasets
    qrels = defaultdict(dict)
    for q in dataset_qrels:
        query_id = q["query-id"]
        corpus_id = q["corpus-id"]
        qrels[query_id][corpus_id] = q["type"]

    queries = {q["_id"]: q["text"] for q in dataset_queries if q["_id"] in qrels.keys()}
    click.echo(f"Loaded {len(queries)} queries")

    # Load the corpus datasets
    datasets_corpus = {}
    for split_name in CoRE[dataset_sub_corpus]:
        dataset_corpus = load_dataset("PaDaS-Lab/CoRE", "corpus")[split_name]
        datasets_corpus[split_name] = dataset_corpus

    # Transform the corpus datasets
    corpora = defaultdict(dict)
    for split_name, dataset_corpus in datasets_corpus.items():
        for d in dataset_corpus:
            corpora[CoRE[dataset_sub_corpus][split_name]][d["_id"]] = {
                "title": d["title"],
                "text": d["text"],
            }
    for corpus_size in corpora:
        click.echo(
            f"Loaded {len(corpora[corpus_size])} documents for corpus of size {corpus_size}"
        )

    # Simplify qrels
    qrels_relevant_only = defaultdict(dict)
    for qid in qrels:
        for docid in qrels[qid]:
            if qrels[qid][docid] == "relevant":
                qrels_relevant_only[qid][docid] = 1

    return corpora, queries, qrels, qrels_relevant_only


def load_data(dataset_name: str, dataset_sub_corpus: str, split: str = "test"):
    """
    Loads the data for the given dataset, i.e., CoRE passage/document or one of the BEIR datasets, from HuggingFace and
    returns the corpus, queries and qrels. If the dataset name is invalid, an error is raised.

    Args:
        dataset_name: The name of the dataset to load.
        dataset_sub_corpus: The specific sub-corpus to load.
        split: The name of the split to load.

    Returns:
        The loaded corpus, queries and qrels.
    """
    if dataset_name == CoRE_NAME:
        return _load_core_data(dataset_sub_corpus)
    elif dataset_name == BEIR_NAME:
        return _load_beir_data(dataset_sub_corpus, split)
    else:
        raise NotImplementedError(
            f"Cannot load data for unsupported dataset {dataset_name}!"
        )


def get_dataset(dataset_name: str) -> List[str] | Dict[str, Dict[str, int]]:
    """
    Returns the dataset dictionary or list containing the names of the datasets to load.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        The list or dictionary containing the dataset names.
    """
    if dataset_name in DATASETS.keys():
        return DATASETS[dataset_name]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported!")
