import os

import click
from tqdm import tqdm
from trectools import TrecPoolMaker, TrecRun

from corect.config import *

RUNS = {
    "passage": [
        "agg-cocondenser",  # 0.3519
        "bm25_splades",  # 0.3848
        "cip_run_1",  # 0.4834
        "cip_run_2",  # 0.4834
        "cip_run_3",  # 0.4688
        "cip_run_4",  # 0.4717
        "cip_run_5",  # 0.4628
        "cip_run_6",  # 0.4670
        "cip_run_7",  # 0.4658
        "naverloo-frgpt4",  # 0.5364
        "naverloo-rgpt4",  # 0.5394
        "naverloo_bm25_RR",  # 0.3844
        "naverloo_bm25_splades_RR",  # 0.4930
        "naverloo_fs",  # 0.4117
        "naverloo_fs_RR",  # 0.4981
        "naverloo_fs_RR_duo",  # 0.5248
        "slim-pp-0shot-uw",  # 0.3574
        "splade_pp_ensemble_distil",  # 0.3818
        "splade_pp_self_distil",  # 0.3858
        "uogtr_b_grf_e",  # 0.3845
        "uogtr_b_grf_e_gb",  # 0.4321
        "uogtr_be",  # 0.3749
        "uogtr_be_gb",  # 0.4236
        # "uogtr_dph",  # 0.2057
        # "uogtr_dph_bo1",  # 0.1311
        "uogtr_qr_be",  # 0.3844
        "uogtr_qr_be_gb",  # 0.4261
        "uogtr_s",  # 0.3817
        "uogtr_se",  # 0.4357
        "uogtr_se_gb",  # 0.4452
        # "uot-yj_LLMs-blender",  # 0.1957
        # "uot-yj_rankgpt35",  # 0.2232
        # "uot-yj_rankgpt4",  # 0.2450
        # "WatS-Augmented-BM25",  # 0.1906
        # "WatS-LLM-Rerank",  # 0.2327
    ],
    "document": [
        # "colbertv2",  # 0.2822
        "D_bm25_splades",  # 0.5120
        "D_naverloo-frgpt4",  # 0.6318
        "D_naverloo_bm25_RR",  # 0.5118
        "D_naverloo_bm_splade_RR",  # 0.5963
    ],
}


def find_distractors(
    queries: dict[str, dict[str, set[str]]],
    all_relevant: set[str],
    num_distractors: int = 10,
    _type: str = "passage",
) -> None:
    """
    Apply repooling by merging the top-k pools of all runs that formed the original judgment pool.
    """
    assert _type in [
        "passage",
        "document",
    ], f"Type {_type} must be either passage or document"

    click.echo(f"Repooling {len(RUNS[_type])} runs:")
    for run in RUNS[_type]:
        click.echo(f"  {run}")
    click.echo()

    # Transform the runs into TrecRun objects
    runs = []
    for run in RUNS[_type]:
        run_path = os.path.join(RESOURCES_FOLDER, "trec32", "deep", f"input.{run}.gz")
        runs.append(TrecRun(run_path))

    # Create a pool from the runs using Reciprocal Rank Fusion (RRF)
    topX = num_distractors + max(
        len([id for v in query_dict["relevant"].values() for id in v])
        for query_dict in queries.values()
    )
    pool = TrecPoolMaker().make_pool(runs, strategy="rrf", topX=topX).pool
    click.echo(
        f"Repooled {len(pool)} queries with on average {len(sum([list(v) for v in pool.values()], [])) / len(pool)} documents each (topX={topX})"
    )

    all_distractors = set()
    for qid, query_dict in tqdm(queries.items()):
        # Check if the query ID is in the pool
        assert qid in pool, f"Query {qid} not in pool"

        # Get the pooled documents for the query
        pool_docs = pool[qid]
        assert isinstance(pool_docs, set), f"Documents for query {qid} are not a set"

        # Get the first N distractors that are not relevant to the query
        distractors = {
            "judged": set(),
            "unjudged": set(),
        }
        distractors_optional = {
            "judged": set(),
            "unjudged": set(),
        }
        for doc in pool_docs:
            relevant = doc in [id for v in query_dict["relevant"].values() for id in v]
            irrelevant = doc in [
                id for v in query_dict["irrelevant"].values() for id in v
            ]
            relevant_to_any = doc in all_relevant

            assert not (
                relevant and irrelevant
            ), f"Document {doc} is both relevant and irrelevant"

            if not relevant:
                # Check if the document is relevant for any other query
                if relevant_to_any:
                    # If the document is relevant for another query, it has to be kept in the final corpus
                    if irrelevant:
                        distractors["judged"].add(doc)
                    else:
                        distractors["unjudged"].add(doc)
                else:
                    # If the document is not relevant for any other query, add it to the optional distractors
                    if irrelevant:
                        distractors_optional["judged"].add(doc)
                    else:
                        distractors_optional["unjudged"].add(doc)

                # Add the distractors to the set of all distractors
                all_distractors.add(doc)

                if len(distractors) + len(distractors_optional) >= num_distractors:
                    break

        # Add the distractors to the query dictionary
        query_dict["distractors"] = distractors
        query_dict["distractors-optional"] = distractors_optional

    click.echo(f"Found {len(all_distractors)} distractors in total")
    return all_distractors
