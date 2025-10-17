import os
import time
from typing import Tuple

import click

from corect.compression_registry import CompressionRegistry
from corect.config import *
from corect.dataset_utils import load_data, get_dataset
from corect.eval_utils import eval_results, search
from corect.model_wrappers import (
    AbstractModelWrapper,
    E5MultilingualWrapper,
    JinaV3Wrapper,
    SnowflakeWrapper
)


def get_model_wrapper(model_name: str) -> Tuple[AbstractModelWrapper, int]:
    """
    Attempts to return the model wrapper used to create embeddings to be evaluated and quantized. If the given model is
    unsupported, an error is thrown instead.

    Args:
        model_name: The model to use.

    Returns:
        The model wrapper, if implemented.
    """
    if model_name == "jinav3":
        return JinaV3Wrapper(), 1024
    elif model_name == "e5":
        return E5MultilingualWrapper(), 1024
    elif model_name == "snowflakev2":
        return SnowflakeWrapper(), 768
    elif model_name == "snowflake":
        return SnowflakeWrapper("Snowflake/snowflake-arctic-embed-m"), 768
    else:
        raise NotImplementedError(f"Model {model_name} not supported!")


@click.command()
@click.argument("model", type=str)
@click.argument("dataset", type=str)
@click.argument("batch_threshold", type=bool, default=True)
def evaluate(model: str, dataset: str, batch_threshold: bool):
    """
    Evaluates the retrieval performance for the given dataset using different quantization methods.

    Args:
        model: A string representing the name of the embedding model (has to be supported by _get_model_wrapper()).
        dataset: A string representing the name of the dataset (has to be supported by get_dataset()).
        batch_threshold: Boolean indicating whether threshold-based quantization methods should compute the thresholds
        per batch.
    """
    embed_model, max_dim = get_model_wrapper(model)
    embed_data = get_dataset(dataset)
    click.echo(f"Loaded model: {embed_model.name}")

    # Initialize compression methods
    CompressionRegistry.add_compressions()

    for data in embed_data:
        click.echo(f"Starting evaluation on dataset {dataset}:{data}")
        embed_folder = os.path.join(EMBED_FOLDER, model, data)
        corpora, queries, qrels, qrels_relevant_only = load_data(dataset, data)

        # Evaluate the dataset
        start = time.time()
        results = search(
            embed_model,
            corpora,
            queries,
            max(K_VALUES),
            CORPUS_CHUNK_SIZE,
            DIMENSIONALITIES,
            embed_folder,
            batch_threshold,
        )
        end = time.time()
        click.echo(f"Time taken: {end - start:.2f} seconds")

        corpus_results = None

        # Loop over corpora
        for corpus_size in sorted(corpora.keys()):
            if corpus_results is None:
                corpus_results = results[corpus_size]
            else:
                for dim in results[corpus_size]:
                    for q in results[corpus_size][dim]:
                        for qid in results[corpus_size][dim][q]:
                            for cid in results[corpus_size][dim][q][qid]:
                                corpus_results[dim][q][qid][cid] = results[corpus_size][dim][q][qid][cid]

            save_path = os.path.join(RESULTS_FOLDER, model, data)
            eval_results(
                DIMENSIONALITIES,
                K_VALUES,
                corpus_size,
                save_path,
                qrels_relevant_only,
                qrels,
                corpus_results,
                max_dim
            )
