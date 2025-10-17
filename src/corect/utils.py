import os
from typing import Dict, Any

import pytrec_eval
import torch

from corect.model_wrappers import AbstractModelWrapper


def count_lines(filepath):
    with open(filepath, "r") as f:
        return sum(1 for _ in f)


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).float()

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b).float()

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def hamming_distance(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the hamming distance between two binary vectors.

    Return:
        Matrix with res[i][j]  = hamming_distance(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).float()

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b).float()

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return -1 * torch.cdist(a, b, p=1)


def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = []

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = {
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        }
        for k in k_values:
            rr = 0
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            MRR[f"MRR@{k}"].append(rr)

    return MRR


def evaluate_results(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

    for k in k_values:
        all_ndcgs[f"NDCG@{k}"] = []
        all_aps[f"MAP@{k}"] = []
        all_recalls[f"Recall@{k}"] = []
        all_precisions[f"P@{k}"] = []

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )
    _mrr = mrr(qrels, results, k_values)

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)
        _mrr[f"MRR@{k}"] = round(sum(_mrr[f"MRR@{k}"]) / len(scores), 5)

    return ndcg, _map, recall, precision, _mrr


def encode_queries(save_path: str, queries: list, encode_kwargs: Dict[str, Any],
                   model: AbstractModelWrapper) -> torch.tensor:
    """
    Encodes the given queries and stores them as pytorch tensors.

    Args:
        save_path: The path where the embeddings are stored.
        queries: The queries to encode.
        encode_kwargs: The keyword arguments passed to the encoding function.
        model: The model used to encode the queries.

    Returns:
        The generated query embeddings.
    """
    query_embeddings = model.encode_queries(
        queries,
        **encode_kwargs,
    )

    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.tensor(query_embeddings)
    else:
        query_embeddings = query_embeddings.detach().cpu()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        torch.save(query_embeddings, os.path.join(save_path, "queries.pt"))

    return query_embeddings


def encode_corpus(save_path: str, corpus: list, encode_kwargs: Dict[str, Any], model: AbstractModelWrapper,
                  batch_num: int):
    """
    Encodes the given corpus documents and stores them as pytorch tensors.

    Args:
        save_path: The path where the embeddings are stored.
        corpus: The documents to encode.
        encode_kwargs: The keyword arguments passed to the encoding function.
        model: The model used to encode the queries.
        batch_num: The number of the current document batch.
    """
    sub_corpus_embeddings = model.encode_corpus(
        corpus,
        **encode_kwargs,
    )

    if not isinstance(sub_corpus_embeddings, torch.Tensor):
        sub_corpus_embeddings = torch.tensor(sub_corpus_embeddings)
    else:
        sub_corpus_embeddings = sub_corpus_embeddings.detach().cpu()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            sub_corpus_embeddings,
            os.path.join(
                save_path, f"corpus_batch_{batch_num}.pt"
            )
        )
