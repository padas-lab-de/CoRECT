import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import torch
from joblib import Parallel, delayed

from corect.compression_registry import CompressionRegistry
from corect.model_wrappers import AbstractModelWrapper
from corect.quantization.AbstractCompression import AbstractCompression
from corect.quantization.BinaryCompression import BinaryCompression
from corect.quantization.LSHCompression import LSHCompression
from corect.quantization.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from corect.quantization.ProductQuantization import ProductQuantization
from corect.utils import cos_sim, evaluate_results, hamming_distance, encode_queries, encode_corpus


def _get_top_k(scores: torch.Tensor, k: int):
    return torch.topk(
        scores.cuda(),
        min(
            k + 1,
            (len(scores[1]) if len(scores) > 1 else len(scores[-1])),
        ),
        dim=1,
        largest=True,
    )


def _load_embeddings(save_path: str, batch_num: int = 0, queries: bool = False) -> Tuple[bool, torch.tensor]:
    """
    Attempts to load query or corpus embeddings at the given path and returns the on success.

    Args:
        save_path: The path where the embeddings are stored.
        batch_num: The number of the batch when loading corpus embeddings.
        queries: Whether to load query or corpus embeddings.

    Returns:
        A boolean indicating whether loading was successful along with the embeddings.
    """
    if save_path:
        try:
            path = os.path.join(save_path, "queries.pt" if queries else f"corpus_batch_{batch_num}.pt")
            embeddings = torch.tensor(torch.load(path, weights_only=False))
            return True, embeddings
        except OSError:
            click.echo(f"Could not find any embeddings at {save_path}.")
    return False, None


def _is_dataset_wide_compression(quantization: AbstractCompression) -> bool:
    """
    Checks if the given quantization is a compression type that is fit on the entire dataset, i.e. PQ or PCA.

    Args:
        quantization: The compression type to check.
    Returns:
        A boolean indicating whether this method needs to be fit to the dataset.
    """
    return (isinstance(quantization, ProductQuantization) or isinstance(quantization, PrincipalComponentAnalysis)
            or isinstance(quantization, LSHCompression))


def _compute_results(dimensionality: int, top_k: int, query_ids: list, result_heaps: dict, compression_methods: list,
                     query_embeddings: torch.tensor) -> defaultdict:
    """
    Computes the top-k retrieved documents per quantization method and query and puts the information into a dictionary
    that contains the name of the quantization method, the query, corpus id and retrieval score. The top-k retrieval
    results computed per batch passed in the results heap serve as the starting point.

    Args:
        dimensionality (int): The dimensionality of the embedding vector.
        top_k (int): The number of documents to retrieve.
        query_ids (list): A list of all query ids.
        result_heaps (dict): The dictionary that holds the initial information on retrieved documents.
        compression_methods (list): The names of the compression methods.
        query_embeddings (torch.tensor): The query embeddings.

    Returns:
        The computed top-k results.
    """
    results = defaultdict()

    for quantization_name in compression_methods:
        if (not quantization_name in result_heaps[dimensionality] or
                not result_heaps[dimensionality][quantization_name]):
            continue
        results[quantization_name] = {}
        top_k_scores, top_k_index = _get_top_k(result_heaps[dimensionality][quantization_name][0], top_k)
        top_k_index = top_k_index.detach().cpu()
        top_k_scores = top_k_scores.detach().cpu()
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            results[quantization_name][query_id] = {}
            corpus_ids = result_heaps[dimensionality][quantization_name][1][query_itr]
            for score, idx in zip(top_k_scores[query_itr], top_k_index[query_itr]):
                corpus_id = corpus_ids[idx]
                results[quantization_name][query_id][corpus_id] = score.item()

    return results


def _get_compression_keys() -> List[str]:
    """
    Returns the list of names of the compression methods under evaluation.

    Returns:
        The list of names.
    """
    quantizations = [key for key in CompressionRegistry.get_compression_methods().keys()]
    if len(CompressionRegistry.get_pretrained_methods()) > 0:
        for key in CompressionRegistry.get_pretrained_methods().keys():
            quantizations.append(key)
    return quantizations



def search(
    model: AbstractModelWrapper,
    corpora: Dict[str, Dict[str, Dict[str, str]]],
    queries: Dict[str, str],
    top_k: int,
    chunk_size: int,
    dimensionalities: List[int],
    save_path: str | None = None,
    batch_threshold: bool = True,
    encode_kwargs: Dict[str, Any] = {},
) -> Dict[Union[str, int], Dict[int, Dict[str, Dict[str, Dict[str, float]]]]]:
    """
    Search for the top-k documents for each query in the corpus.
    """
    # Load compression methods
    compression_methods = CompressionRegistry.get_compression_methods()

    # Initialize results dict
    query_ids = list(queries.keys())
    corpus_sizes = sorted(corpora.keys())
    results = {}
    result_heaps = defaultdict(dict)
    dimensionalities = sorted(dimensionalities, reverse=True)
    for dimensionality in dimensionalities:
        for quantization_name, quantization in compression_methods.items():
            if _is_dataset_wide_compression(quantization) and dimensionality != max(dimensionalities):
                continue
            for corpus_size in corpus_sizes:
                results[corpus_size] = results.get(corpus_size, defaultdict(dict))
                results[corpus_size][dimensionality][quantization_name] = {
                    qid: {} for qid in query_ids
                }

            # Initialize one heaps dict for all corpus sizes
            result_heaps[dimensionality][quantization_name] = None

    # Embed queries or load saved embeddings:
    loaded_queries, query_embeddings = _load_embeddings(save_path, 0, True)

    if loaded_queries:
        click.echo(f"Loaded query embeddings at {save_path}")
    else:
        with torch.no_grad():
            queries = [queries[qid] for qid in queries]
            query_embeddings = encode_queries(save_path, queries, encode_kwargs, model)

    # Loop over corpora
    for corpus_size in corpus_sizes:
        click.echo(f"Evaluating corpus of size {corpus_size}...")

        # Embed corpus
        corpus = corpora[corpus_size]
        corpus_ids = list(corpus.keys())
        corpus_ids = sorted(corpus_ids)
        corpus = [corpus[cid] for cid in corpus_ids]

        # Encoding corpus in batches... Warning: This might take a while!
        iterator = range(0, len(corpus), chunk_size)

        for batch_num, corpus_start_idx in enumerate(iterator):
            if not os.path.exists(os.path.join(save_path, str(corpus_size), f"corpus_batch_{batch_num}.pt")):
                click.echo(f"Encoding Batch {batch_num + 1}/{len(iterator)}...")
                corpus_end_idx = min(corpus_start_idx + chunk_size, len(corpus))
                with torch.no_grad():
                    # Encode chunk of corpus
                    encode_corpus(os.path.join(save_path, str(corpus_size)), corpus[corpus_start_idx:corpus_end_idx],
                                  encode_kwargs, model, batch_num)
        torch.cuda.empty_cache()

        # Determine global quantization boundaries for dataset
        if not batch_threshold:
            click.echo("Computing dataset-wide thresholds")
            corpus_embeddings = []
            num_batches = 0
            for size in corpus_sizes:
                if isinstance(size, int) and size > corpus_size:
                    break
                corpus_iterator = range(0, len(corpora[size]), chunk_size)
                for batch_num in range(len(corpus_iterator)):
                    # If the dataset has more than 1M documents, compute boundaries only on the first batches
                    if num_batches * chunk_size >= 1000000:
                        break
                    loaded, embeddings = _load_embeddings(os.path.join(save_path, str(size)), batch_num)
                    corpus_embeddings.append(embeddings)
                    num_batches += 1
            corpus_embeddings = torch.vstack(corpus_embeddings)
            for quantization in compression_methods.values():
                # Compute per dataset threshold for threshold-based methods
                if hasattr(quantization, "compute_threshold"):
                    quantization.compute_threshold(corpus_embeddings)
                elif _is_dataset_wide_compression(quantization):
                    quantization.fit(corpus_embeddings)

        # Quantize and evaluate retrieval performance
        for batch_num, corpus_start_idx in enumerate(iterator):
            loaded_corpus, sub_corpus_embeddings = _load_embeddings(os.path.join(save_path, str(corpus_size)),
                                                                    batch_num)

            if loaded_corpus:
                click.echo(
                    f"Loaded corpus embeddings for batch {batch_num + 1}/{len(iterator)} at {save_path}"
                )
            else:
                raise FileNotFoundError(f"Could not load corpus embeddings for batch {batch_num + 1}/{len(iterator)}!")

            # Loop over all quantizations and dimensions
            for quantization_name, quantization in compression_methods.items():
                query_embeds, corpus_embeds = None, None
                for dimensionality in dimensionalities:
                    if _is_dataset_wide_compression(quantization) and dimensionality != max(dimensionalities):
                        continue
                    # Quantize the embeddings
                    if query_embeds is None or corpus_embeds is None:
                        corpus_embeds = quantization.compress(
                            sub_corpus_embeddings[:, :dimensionality]
                        )
                        query_embeds = quantization.compress(
                            query_embeddings[:, :dimensionality]
                        )

                    _query_embeddings = query_embeds[:, :dimensionality].detach().clone()
                    _sub_corpus_embeddings = corpus_embeds[:, :dimensionality].detach().clone()

                    # Use cosine similarity for quantized embeddings, hamming distance for binarized
                    if not isinstance(quantization, BinaryCompression):
                        similarity_scores = cos_sim(
                            _query_embeddings, _sub_corpus_embeddings
                        ).detach().cpu()
                    else:
                        similarity_scores = hamming_distance(
                            _query_embeddings, _sub_corpus_embeddings
                        ).detach().cpu()

                    # Check for NaN values
                    assert torch.isnan(similarity_scores).sum() == 0

                    # Get top-k values
                    similarity_scores_top_k_values, similarity_scores_top_k_idx = _get_top_k(similarity_scores, top_k)
                    similarity_scores_top_k_values = (
                        similarity_scores_top_k_values.detach().cpu()
                    )
                    similarity_scores_top_k_idx = (
                        similarity_scores_top_k_idx.detach().cpu() + corpus_start_idx
                    )
                    top_k_corpus_ids = []
                    for top_k_indices in similarity_scores_top_k_idx.tolist():
                        top_k_corpus_ids.append([corpus_ids[idx] for idx in top_k_indices])

                    # Concatenate top-k values and indices from different batches
                    if result_heaps[dimensionality][quantization_name] is None:
                        result_heaps[dimensionality][quantization_name] = (
                            similarity_scores_top_k_values, top_k_corpus_ids
                        )
                    else:
                        prev_top_k = result_heaps[dimensionality][quantization_name][1]
                        for idx, top_k_ids in enumerate(top_k_corpus_ids):
                            prev_top_k[idx] += top_k_ids
                        result_heaps[dimensionality][quantization_name] = (
                            torch.cat((result_heaps[dimensionality][quantization_name][0],
                                       similarity_scores_top_k_values), 1),
                            prev_top_k
                        )

                    # Reduce tracked top-k values every 50 batches
                    if batch_num > 0 and batch_num % 50 == 0:
                        top_k_scores, top_k_index = _get_top_k(result_heaps[dimensionality][quantization_name][0],
                                                               top_k)
                        top_k_index = top_k_index.detach().cpu()
                        indices = []
                        for idx in range(len(result_heaps[dimensionality][quantization_name][1])):
                            cids = []
                            for top_k_idx in top_k_index[idx]:
                                cids.append(result_heaps[dimensionality][quantization_name][1][idx][top_k_idx])
                            indices.append(cids)
                        result_heaps[dimensionality][quantization_name] = (top_k_scores.detach().cpu(), indices)

        res = Parallel(n_jobs=6)(delayed(_compute_results)(dim, top_k, query_ids, result_heaps,
                                                           _get_compression_keys(), query_embeddings) for dim in
                                 dimensionalities)
        assert len(res) == len(dimensionalities)
        for idx, dimensionality in enumerate(dimensionalities):
            results[corpus_size][dimensionality] = res[idx]

    torch.cuda.empty_cache()
    return results


def rc(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> Dict[str, float]:
    """
    Compute the Relevance Composition (RC) score for each query.

    Relevance composition @ k yields the proportion of retrieved documents that are
    relevant, distractors, and randoms among the top-ranked results.
    """
    RC = {}

    k_max, top_hits = max(k_values), {}
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    query_relevant_docs = defaultdict(set)
    query_distractor_docs = defaultdict(set)
    for query_id in qrels:
        for doc_id in qrels[query_id]:
            if qrels[query_id][doc_id] == "relevant":
                query_relevant_docs[query_id].add(doc_id)
            elif qrels[query_id][doc_id] == "distractor":
                query_distractor_docs[query_id].add(doc_id)

    for k in k_values:
        relevant_count = 0
        distractor_count = 0
        for query_id in top_hits:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs[query_id]:
                    relevant_count += 1
                elif hit[0] in query_distractor_docs[query_id]:
                    distractor_count += 1
        RC[f"RC@{k}"] = {
            "relevant": round(relevant_count / len(top_hits), 5),
            "distractor": round(distractor_count / len(top_hits), 5),
        }

    return RC


def _get_rankings(
    results: dict, qrels_rel: defaultdict, qrels: defaultdict, top_k: int
) -> Dict[Any, Dict[str, Dict[str, int]]]:
    """
    Stores the position of all relevant, random and distractor documents retrieved in the top-k results for a certain
    query in a dictionary.

    Args:
        results: Dictionary containing the results of the top-k retrieval.
        qrels_rel: The qrels containing only relevant documents.
        qrels: The qrels that also contain information on distractors.

    Returns:
        The constructed dictionary.
    """
    rankings = {}
    for qid in results:
        rankings[qid] = {
            "relevant": {},
            "distractor": {},
            "random": {},
        }
        scores = {}

        for cid in results[qid]:
            scores[cid] = results[qid][cid]

        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

        for idx, cid in enumerate(scores):
            if idx >= top_k:
                break
            if cid in qrels_rel[qid] and qrels_rel[qid][cid] > 0:
                rankings[qid]["relevant"][cid] = idx + 1
            elif cid in qrels[qid] and qrels[qid][cid] == "distractor":
                rankings[qid]["distractor"][cid] = idx + 1
            else:
                rankings[qid]["random"][cid] = idx + 1

    return rankings


def _compute_metrics(k_values: List[int], qrels_relevant_only: defaultdict, qrels: Optional[defaultdict],
                     results: dict):
    """
    Calculates retrieval metrics at different k values for the given results.

    Args:
        k_values: The k values for which to compute the metrics.
        qrels_relevant_only: The qrels containing only relevant documents.
        qrels: The qrels that also contain information on distractors.
        results: The retrieval results per query.

    Returns:
        A dictionary containing the calculated metrics.
    """
    ndcg, _map, recall, precision, mrr = evaluate_results(
        qrels_relevant_only.copy(),
        results,
        k_values,
    )

    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{
            f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()
        },
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
    }

    rankings = _get_rankings(results.copy(), qrels_relevant_only.copy(), qrels.copy(), max(k_values))
    scores["rankings"] = rankings

    # Custom evaluation
    if qrels:
        _rc = rc(
            qrels,
            results,
            k_values,
        )
        for k, v in _rc.items():
            scores[f"rc_at_{k.split('@')[1]}"] = v
    return scores


def eval_results(
    dimensionalities: List[int],
    k_values: List[int],
    corpus_size: int | str,
    save_path: str,
    qrels_relevant_only: defaultdict,
    qrels: defaultdict,
    results: dict,
    max_dim: int,
):
    """
    Evaluates dense retrieval performance across different embedding vector lengths and compression methods. The results
    are then stored in a directory structure under the specified save directory split by embedding dimensionality and
    quantization method.

    Args:
        dimensionalities: The different embedding vector lengths.
        k_values: A list of k values for which metrics should be computed.
        corpus_size: The corpus size (relevant when using CoRE, defaults to the dataset name for BEIR).
        save_path: The directory to save the results.
        qrels_relevant_only: The qrels containing only relevant documents.
        qrels: The qrels that also contain information on distractors.
        results: The results of the retrieval.
        max_dim: The model's maximum dimensionality.
    """
    for dimensionality in dimensionalities:
        for quantization in _get_compression_keys():
            if quantization in results[dimensionality]:
                click.echo(
                    f"Evaluating corpus {corpus_size}, dimensionality {dimensionality} and quantization {quantization}"
                )

                # Evaluate the results
                scores = _compute_metrics(k_values, qrels_relevant_only, qrels, results[dimensionality][quantization])
                click.echo(f"NDCG@10: {scores['ndcg_at_10']}")
                dim_dir = max_dim if dimensionality > max_dim else dimensionality

                # Save the results
                results_path = os.path.join(
                    save_path,
                    f"dim={dim_dir}",
                    f"q={quantization}",
                    f"{corpus_size}.json",
                )
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(scores, f, indent=4)
