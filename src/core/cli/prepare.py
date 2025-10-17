import os
from collections import defaultdict

import click
import pandas as pd

from core.config import *


@click.command()
@click.argument("upload_id", type=str)
def prepare(upload_id: str):
    """
    Create collection of datasets varying in complexity.
    """
    click.echo("Creating dataset collection...")

    # Download here: https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt
    qrels_passage_path = os.path.join(
        DATASETS_FOLDER, "downloads", "2023.qrels.pass.withDupes.txt"
    )

    # Download here: https://trec.nist.gov/data/deep/2023.qrels.docs.withDupes.txt
    qrels_document_path = os.path.join(
        DATASETS_FOLDER, "downloads", "2023.qrels.docs.withDupes.txt"
    )

    # Download here: https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv
    queries_path = os.path.join(DATASETS_FOLDER, "downloads", "2023_queries.tsv")

    # Load passage qrels
    qrels_passage = defaultdict(list)
    with open(qrels_passage_path, "r") as file:
        for line in file:
            qid, _, pid, rel = line.strip().split()
            qrels_passage["qid"].append(qid)
            qrels_passage["pid"].append(pid)
            qrels_passage["score"].append(int(rel))

    click.echo(
        f"Loaded {len(qrels_passage['qid'])} passage qrels for {len(set(qrels_passage['qid']))} qids and {len(set(qrels_passage['pid']))} pids"
    )

    # Load document qrels
    qrels_document = defaultdict(list)
    with open(qrels_document_path, "r") as file:
        for line in file:
            qid, _, docid, rel = line.strip().split()
            qrels_document["qid"].append(qid)
            qrels_document["docid"].append(docid)
            qrels_document["score"].append(int(rel))

    # Load queries
    queries = {}
    with open(queries_path, "r") as file:
        for line in file:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    click.echo(f"Loaded {len(queries)} queries")

    # Keep only those queries that are in the qrels
    queries = {
        k: v
        for k, v in queries.items()
        if k in set(qrels_passage["qid"]) and k in set(qrels_document["qid"])
    }
    click.echo(f"Remaining {len(queries)} queries after removing those not in qrels")

    # Create HuggingFace dataset
    datasets_dict = defaultdict(dict)
    datasets_dict["qrels-passage"]["qid"] = [q for q in qrels_passage["qid"]]
    datasets_dict["qrels-passage"]["pid"] = [q for q in qrels_passage["pid"]]
    datasets_dict["qrels-passage"]["score"] = [q for q in qrels_passage["score"]]
    datasets_dict["qrels-document"]["qid"] = [q for q in qrels_document["qid"]]
    datasets_dict["qrels-document"]["docid"] = [q for q in qrels_document["docid"]]
    datasets_dict["qrels-document"]["score"] = [q for q in qrels_document["score"]]
    datasets_dict["queries"]["_id"] = list(queries.keys())
    datasets_dict["queries"]["text"] = list(queries.values())

    # Save datasets to disk
    temp_qrels_passage_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "passage.jsonl"
    )
    temp_qrels_document_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "document.jsonl"
    )
    temp_queries_path = os.path.join(TEMP_FOLDER, upload_id, "queries.jsonl")
    os.makedirs(os.path.dirname(temp_qrels_passage_path), exist_ok=True)

    pd.DataFrame(datasets_dict["qrels-passage"]).to_json(
        temp_qrels_passage_path, orient="records", lines=True
    )
    click.echo(f"Saved passage qrels to {temp_qrels_passage_path}")
    pd.DataFrame(datasets_dict["qrels-document"]).to_json(
        temp_qrels_document_path, orient="records", lines=True
    )
    click.echo(f"Saved document qrels to {temp_qrels_document_path}")
    pd.DataFrame(datasets_dict["queries"]).to_json(
        temp_queries_path, orient="records", lines=True
    )
    click.echo(f"Saved queries to {temp_queries_path}")
