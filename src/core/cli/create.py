import gzip
import json
import os
import pickle
import random
from collections import defaultdict

import click
import pandas as pd

from core.cli.sampling import find_distractors
from core.config import *


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


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
NUM_RELEVANT_PER_QUERY = 10
NUM_DISTRACTORS_PER_QUERY = 100


def _load_pids_dict() -> dict[str, str]:
    """
    Load the pids_dict.pkl file.
    """
    # Download here: https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar
    passage_path = os.path.join(
        DATASETS_FOLDER, "downloads", "msmarco_v2_passage", "msmarco_passage_{i:02}.gz"
    )

    pids_dict_path = os.path.join(DATASETS_FOLDER, "init", "pids_dict.pkl")
    os.makedirs(os.path.dirname(pids_dict_path), exist_ok=True)
    if os.path.exists(pids_dict_path):
        with open(pids_dict_path, "rb") as file:
            pids_dict = pickle.load(file)
        click.echo(f"Loaded {len(pids_dict)} pids with docids")

    else:
        click.echo(f"Creating pids dict from passage files...")
        pids_dict = {}
        for i in range(0, 70):
            click.echo(f"Passage: {i}")
            current_passage_path = passage_path.format(i=i)

            with gzip.open(current_passage_path, "rt", encoding="utf-8") as file:

                for line in file:
                    try:
                        passage_json = json.loads(line)
                        pid = passage_json["pid"]
                        docid = passage_json["docid"]
                    except Exception as e:
                        click.echo(
                            f"Skipping corrupt line in passage file {i}: {e}", err=True
                        )
                        continue

                    pids_dict[pid] = docid

        # Save list of pids with docids
        click.echo(f"Loaded {len(pids_dict)} pids with docids in total")
        with open(pids_dict_path, "wb") as file:
            pickle.dump(pids_dict, file)
        click.echo(f"Saved pids with docids to {pids_dict_path}")

    return pids_dict


def _prepare_corpus_passage(
    entire_corpus_passage_set: set[str],
) -> dict[str, dict[str, str]]:
    """
    Extract the downloaded files and merge passages with documents.
    """
    entire_corpus_passage = {}

    # Download here: https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar
    passage_path = os.path.join(
        DATASETS_FOLDER, "downloads", "msmarco_v2_passage", "msmarco_passage_{i:02}.gz"
    )

    # Extract gzipped passage files and merge with doc files
    for i in range(0, 70):
        click.echo(f"Passage: {i}")
        current_passage_path = passage_path.format(i=i)

        with gzip.open(current_passage_path, "rt", encoding="utf-8") as file:

            for line in file:
                try:
                    passage_json = json.loads(line)
                    pid = passage_json["pid"]
                except Exception as e:
                    click.echo(
                        f"Skipping corrupt line in passage file {i}: {e}", err=True
                    )
                    continue

                # Check if the pid is in the entire corpus set
                if pid not in entire_corpus_passage_set:
                    continue

                entire_corpus_passage[pid] = {
                    "title": "",
                    "headings": "",
                    "text": passage_json["passage"],
                }

    return entire_corpus_passage


def _prepare_corpus_document(
    entire_corpus_document_set: set[str],
) -> dict[str, dict[str, str]]:
    """
    Extract the downloaded files and merge passages with documents.
    """
    entire_corpus_document = {}

    # Download here: https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_doc.tar
    doc_path = os.path.join(
        DATASETS_FOLDER, "downloads", "msmarco_v2_doc", "msmarco_doc_{i:02}.gz"
    )

    # Extract gzipped doc files
    for i in range(0, 60):
        click.echo(f"Document: {i}")
        current_doc_path = doc_path.format(i=i)

        with gzip.open(current_doc_path, "rt", encoding="utf-8") as file:
            for line in file:
                try:
                    document_json = json.loads(line)
                    docid = document_json["docid"]
                except Exception as e:
                    click.echo(f"Skipping corrupt line in doc file {i}: {e}", err=True)
                    continue

                # Check if the docid is in the entire corpus set
                if docid not in entire_corpus_document_set:
                    continue

                entire_corpus_document[docid] = {
                    "title": document_json["title"],
                    "headings": document_json["headings"],
                    "text": document_json["body"],
                }

    return entire_corpus_document


@click.command()
@click.argument("upload_id", type=str)
def create(upload_id: str):
    """
    Create collection of datasets varying in complexity.
    """
    click.echo("Creating dataset collection...")

    # Random seed
    random.seed(42)

    ################################ Passages ################################

    # Load passage qrels
    temp_qrels_passage_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "passage.jsonl"
    )
    assert os.path.exists(
        temp_qrels_passage_path
    ), f"qrels/passage.jsonl not found in {TEMP_FOLDER}/{upload_id}"
    df_qrels_passage = pd.read_json(temp_qrels_passage_path, lines=True)
    click.echo(f"Loaded {len(df_qrels_passage)} passage qrels")

    # Initialize queries dictionary for passage qrels
    all_relevant_passage = set()
    queries_dict_passage = {}
    for _, row in df_qrels_passage.iterrows():
        qid = row["qid"]
        pid = row["pid"]
        # docid = row["docid"]
        score = row["score"]

        if qid not in queries_dict_passage:
            queries_dict_passage[qid] = {}
            queries_dict_passage[qid]["relevant"] = defaultdict(set)
            queries_dict_passage[qid]["irrelevant"] = defaultdict(set)

        if score >= 2:
            queries_dict_passage[qid]["relevant"][score].add(pid)
            all_relevant_passage.add(pid)
        else:
            queries_dict_passage[qid]["irrelevant"][score].add(pid)
    click.echo(
        f"Loaded {len(queries_dict_passage)} queries with {len(all_relevant_passage)} relevant passages in total"
    )

    # Find distractors for passage qrels
    all_distractors_passage = find_distractors(
        queries_dict_passage, all_relevant_passage, num_distractors=100, _type="passage"
    )

    # Output for debugging
    debug_file = os.path.join(TEMP_FOLDER, upload_id, "debug-passage.json")
    with open(debug_file, "w") as file:
        file.write(json.dumps(queries_dict_passage, indent=4, cls=SetEncoder))

    ################################ Documents ################################

    # Load document qrels
    temp_qrels_document_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "document.jsonl"
    )
    assert os.path.exists(
        temp_qrels_document_path
    ), f"qrels/document.jsonl not found in {TEMP_FOLDER}/{upload_id}"
    df_qrels_document = pd.read_json(temp_qrels_document_path, lines=True)
    click.echo(f"Loaded {len(df_qrels_document)} document qrels")

    # Initialize queries dictionary for document qrels
    all_relevant_document = set()
    queries_dict_document = {}
    for _, row in df_qrels_document.iterrows():
        qid = row["qid"]
        docid = row["docid"]
        score = row["score"]

        if qid not in queries_dict_document:
            queries_dict_document[qid] = {}
            queries_dict_document[qid]["relevant"] = defaultdict(set)
            queries_dict_document[qid]["irrelevant"] = defaultdict(set)

        if score >= 1:
            queries_dict_document[qid]["relevant"][score].add(docid)
            all_relevant_document.add(docid)
        else:
            queries_dict_document[qid]["irrelevant"][score].add(docid)
    click.echo(
        f"Loaded {len(queries_dict_document)} queries with {len(all_relevant_document)} relevant documents in total"
    )

    # Find all distractors for document qrels
    all_distractors_document = find_distractors(
        queries_dict_document,
        all_relevant_document,
        num_distractors=100,
        _type="document",
    )

    # Output for debugging
    debug_file = os.path.join(TEMP_FOLDER, upload_id, "debug-document.json")
    with open(debug_file, "w") as file:
        file.write(json.dumps(queries_dict_document, indent=4, cls=SetEncoder))

    ################################ Passages ################################

    # Construct corpus and qrels
    corpus_passage = {
        "core": set(),
    }
    qrels_passage = {}

    for qid, query_dict in queries_dict_passage.items():
        current_relevant_docs = set()

        # First: Add relevant documents with score 3
        if 3 in query_dict["relevant"]:
            relevant_3 = sorted(query_dict["relevant"][3])
            i = min(NUM_RELEVANT_PER_QUERY, len(relevant_3))
            current_relevant_docs.update(relevant_3[:i])
            click.echo(
                f"  Added {i} relevant documents with score 3 to query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )

        # Second: Add relevant documents with score 2
        if (
            len(current_relevant_docs) < NUM_RELEVANT_PER_QUERY
            and 2 in query_dict["relevant"]
        ):
            relevant_2 = sorted(query_dict["relevant"][2])
            i = min(
                NUM_RELEVANT_PER_QUERY - len(current_relevant_docs), len(relevant_2)
            )
            current_relevant_docs.update(relevant_2[:i])
            click.echo(
                f"  Added {i} relevant documents with score 2 to query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )

        # Check if we have enough relevant documents
        if len(current_relevant_docs) < NUM_RELEVANT_PER_QUERY:
            click.echo(
                f"  Not enough relevant documents for query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )
            continue

        current_distractors = set()

        # Check the number of distractors already in the corpus
        for pid in corpus_passage["core"]:
            if (
                pid in query_dict["distractors"]["judged"]
                or pid in query_dict["distractors"]["unjudged"]
            ):
                current_distractors.add(pid)
        click.echo(
            f"  Found {len(current_distractors)} distractors in the corpus for query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
        )

        # First: Add judged optional distractors
        distractors_optional_judged = sorted(
            query_dict["distractors-optional"]["judged"]
        )
        if len(distractors_optional_judged) > 0:
            i = min(
                NUM_DISTRACTORS_PER_QUERY - len(current_distractors),
                len(distractors_optional_judged),
            )
            current_distractors.update(distractors_optional_judged[:i])
            click.echo(
                f"  Added {i} judged optional distractors to query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
            )

        # Second: Add unjudged optional distractors
        distractors_optional_unjudged = sorted(
            query_dict["distractors-optional"]["unjudged"]
        )
        if (
            len(current_distractors) < NUM_DISTRACTORS_PER_QUERY
            and len(distractors_optional_unjudged) > 0
        ):
            i = min(
                NUM_DISTRACTORS_PER_QUERY - len(current_distractors),
                len(distractors_optional_unjudged),
            )
            current_distractors.update(distractors_optional_unjudged[:i])
            click.echo(
                f"  Added {i} unjudged optional distractors to query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
            )

        # Check if we have enough optional distractors
        if len(current_distractors) < NUM_DISTRACTORS_PER_QUERY:
            click.echo(
                f"  Not enough optional distractors for query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
            )
            continue

        # Add query to qrels
        click.echo(f"  Adding query {qid} to corpus")
        qrels_passage[qid] = {
            "relevant": [],
            "distractor": [],
        }

        # Add relevant documents to corpus
        for pid in sorted(current_relevant_docs):
            corpus_passage["core"].add(pid)
            qrels_passage[qid]["relevant"].append(pid)

        # Add distractors to corpus
        for pid in sorted(current_distractors):
            corpus_passage["core"].add(pid)
            qrels_passage[qid]["distractor"].append(pid)

    click.echo(f"Created passage qrels with {len(qrels_passage)} queries")

    # Create HuggingFace dataset
    datasets_dict = {}
    datasets_dict["qrels-passage"] = defaultdict(list)
    for qid in qrels_passage:
        for pid in qrels_passage[qid]["relevant"]:
            datasets_dict["qrels-passage"]["query-id"].append(qid)
            datasets_dict["qrels-passage"]["corpus-id"].append(pid)
            datasets_dict["qrels-passage"]["type"].append("relevant")
        for pid in qrels_passage[qid]["distractor"]:
            datasets_dict["qrels-passage"]["query-id"].append(qid)
            datasets_dict["qrels-passage"]["corpus-id"].append(pid)
            datasets_dict["qrels-passage"]["type"].append("distractor")

    # Save datasets to disk
    temp_qrels_passage_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "passage-final.jsonl"
    )

    pd.DataFrame(datasets_dict["qrels-passage"]).to_json(
        temp_qrels_passage_path, orient="records", lines=True
    )
    click.echo(f"Saved passage qrels to {temp_qrels_passage_path}")

    ################################ Documents ################################

    # Construct corpus and qrels
    corpus_document = {
        "core": set(),
    }
    qrels_document = {}

    for qid, query_dict in queries_dict_document.items():
        current_relevant_docs = set()

        # First: Add relevant documents with score 3
        if 3 in query_dict["relevant"]:
            relevant_3 = sorted(query_dict["relevant"][3])
            i = min(NUM_RELEVANT_PER_QUERY, len(relevant_3))
            current_relevant_docs.update(relevant_3[:i])
            click.echo(
                f"  Added {i} relevant documents with score 3 to query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )

        # Second: Add relevant documents with score 2
        if (
            len(current_relevant_docs) < NUM_RELEVANT_PER_QUERY
            and 2 in query_dict["relevant"]
        ):
            relevant_2 = sorted(query_dict["relevant"][2])
            i = min(
                NUM_RELEVANT_PER_QUERY - len(current_relevant_docs), len(relevant_2)
            )
            current_relevant_docs.update(relevant_2[:i])
            click.echo(
                f"  Added {i} relevant documents with score 2 to query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )

        # Third: Add relevant documents with score 1
        if (
            len(current_relevant_docs) < NUM_RELEVANT_PER_QUERY
            and 1 in query_dict["relevant"]
        ):
            relevant_1 = sorted(query_dict["relevant"][1])
            i = min(
                NUM_RELEVANT_PER_QUERY - len(current_relevant_docs), len(relevant_1)
            )
            current_relevant_docs.update(relevant_1[:i])
            click.echo(
                f"  Added {i} relevant documents with score 1 to query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )

        # Check if we have enough relevant documents
        if len(current_relevant_docs) < NUM_RELEVANT_PER_QUERY:
            click.echo(
                f"  Not enough relevant documents for query {qid} (current: {len(current_relevant_docs)}/{NUM_RELEVANT_PER_QUERY})"
            )
            continue

        current_distractors = set()

        # Check the number of distractors already in the corpus
        for docid in corpus_document["core"]:
            if (
                docid in query_dict["distractors"]["judged"]
                or docid in query_dict["distractors"]["unjudged"]
            ):
                current_distractors.add(docid)
        click.echo(
            f"  Found {len(current_distractors)} distractors in the corpus for query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
        )

        # First: Add judged optional distractors
        distractors_optional_judged = sorted(
            query_dict["distractors-optional"]["judged"]
        )
        if len(distractors_optional_judged) > 0:
            i = min(
                NUM_DISTRACTORS_PER_QUERY - len(current_distractors),
                len(distractors_optional_judged),
            )
            current_distractors.update(distractors_optional_judged[:i])
            click.echo(
                f"  Added {i} judged optional distractors to query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
            )

        # Second: Add unjudged optional distractors
        distractors_optional_unjudged = sorted(
            query_dict["distractors-optional"]["unjudged"]
        )
        if (
            len(current_distractors) < NUM_DISTRACTORS_PER_QUERY
            and len(distractors_optional_unjudged) > 0
        ):
            i = min(
                NUM_DISTRACTORS_PER_QUERY - len(current_distractors),
                len(distractors_optional_unjudged),
            )
            current_distractors.update(distractors_optional_unjudged[:i])
            click.echo(
                f"  Added {i} unjudged optional distractors to query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
            )

        # Check if we have enough optional distractors
        if len(current_distractors) < NUM_DISTRACTORS_PER_QUERY:
            click.echo(
                f"  Not enough optional distractors for query {qid} (current: {len(current_distractors)}/{NUM_DISTRACTORS_PER_QUERY})"
            )
            continue

        # Add query to qrels
        click.echo(f"  Adding query {qid} to corpus")
        qrels_document[qid] = {
            "relevant": [],
            "distractor": [],
        }

        # Add relevant documents to corpus
        for docid in sorted(current_relevant_docs):
            corpus_document["core"].add(docid)
            qrels_document[qid]["relevant"].append(docid)

        # Add distractors to corpus
        for docid in sorted(current_distractors):
            corpus_document["core"].add(docid)
            qrels_document[qid]["distractor"].append(docid)

    click.echo(f"Created document qrels with {len(qrels_document)} queries")

    # Create HuggingFace dataset
    datasets_dict = {}
    datasets_dict["qrels-document"] = defaultdict(list)
    for qid in qrels_document:
        for docid in qrels_document[qid]["relevant"]:
            datasets_dict["qrels-document"]["query-id"].append(qid)
            datasets_dict["qrels-document"]["corpus-id"].append(docid)
            datasets_dict["qrels-document"]["type"].append("relevant")
        for docid in qrels_document[qid]["distractor"]:
            datasets_dict["qrels-document"]["query-id"].append(qid)
            datasets_dict["qrels-document"]["corpus-id"].append(docid)
            datasets_dict["qrels-document"]["type"].append("distractor")

    # Save datasets to disk
    temp_qrels_document_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "document-final.jsonl"
    )

    pd.DataFrame(datasets_dict["qrels-document"]).to_json(
        temp_qrels_document_path, orient="records", lines=True
    )
    click.echo(f"Saved document qrels to {temp_qrels_document_path}")

    ################################ Queries ################################

    qids = set()
    qids.update(qrels_passage.keys())
    qids.update(qrels_document.keys())

    # Load queries
    temp_queries_path = os.path.join(TEMP_FOLDER, upload_id, "queries.jsonl")
    assert os.path.exists(
        temp_queries_path
    ), f"queries.jsonl not found in {TEMP_FOLDER}/{upload_id}"
    df_queries = pd.read_json(temp_queries_path, lines=True)
    click.echo(f"Loaded {len(df_queries)} queries")

    # Keep only those queries that are in the qrels
    queries_dict = {}
    for _, row in df_queries.iterrows():
        qid = row["_id"]
        query = row["text"]

        if qid not in qids:
            continue

        queries_dict[qid] = query
    click.echo(
        f"Remaining {len(queries_dict)} queries after removing those not in qrels"
    )

    # Create HuggingFace dataset
    datasets_dict = {}
    datasets_dict["queries"] = defaultdict(list)
    for qid in queries_dict:
        datasets_dict["queries"]["_id"].append(qid)
        datasets_dict["queries"]["text"].append(queries_dict[qid])

    # Save datasets to disk
    temp_queries_path = os.path.join(TEMP_FOLDER, upload_id, "queries-final.jsonl")
    pd.DataFrame(datasets_dict["queries"]).to_json(
        temp_queries_path, orient="records", lines=True
    )

    ################################ Random ################################

    # Loading pids with docids
    pids_dict = _load_pids_dict()

    # Loading random documents
    random_passages = set()
    random_documents = set()
    for pid, docid in pids_dict.items():
        # if pid not in all_relevant_passage \
        #         and pid not in all_distractors_passage:
        #     random_passages.add(pid)

        # if docid not in all_relevant_document \
        #         and docid not in all_distractors_document:
        #     random_documents.add(docid)
        if (
            pid not in all_relevant_passage
            and pid not in all_distractors_passage
            and docid not in all_relevant_document
            and docid not in all_distractors_document
        ):
            random_passages.add(pid)
            random_documents.add(docid)
    click.echo(
        f"Loaded {len(random_passages)} random passages and {len(random_documents)} random documents"
    )
    del pids_dict

    # Fill passage corpus up with random documents
    current_dataset_size = len(corpus_passage["core"])
    for dataset_size in sorted(DATASETS["passage"]):
        i = max(0, dataset_size - current_dataset_size)
        current_dataset_size = dataset_size

        # Add random documents to corpus
        corpus_passage[dataset_size] = random.sample(sorted(random_passages), i)
        random_passages = set(random_passages) - set(corpus_passage[dataset_size])
        click.echo(
            f"Added {len(corpus_passage[dataset_size])} random passages to dataset {dataset_size}"
        )
    del random_passages

    # Fill document corpus up with random documents
    current_dataset_size = len(corpus_document["core"])
    for dataset_size in sorted(DATASETS["document"]):
        i = max(0, dataset_size - current_dataset_size)
        current_dataset_size = dataset_size

        # Add random documents to corpus
        corpus_document[dataset_size] = random.sample(sorted(random_documents), i)
        random_documents = set(random_documents) - set(corpus_document[dataset_size])
        click.echo(
            f"Added {len(corpus_document[dataset_size])} random documents to dataset {dataset_size}"
        )
    del random_documents

    ################################ Passages ################################

    # Fetch entire corpus of passages
    entire_corpus_passage_set = set()
    for k in ["core"] + DATASETS["passage"]:
        if k not in corpus_passage:
            continue
        for pid in corpus_passage[k]:
            assert (
                pid not in entire_corpus_passage_set
            ), f"Duplicate pid {pid} found in corpus"
            entire_corpus_passage_set.add(pid)

    # Load entire corpus of passages
    entire_corpus_passage = _prepare_corpus_passage(entire_corpus_passage_set)
    click.echo(f"Loaded {len(entire_corpus_passage)} passages")

    for k in ["core"] + DATASETS["passage"]:
        if k not in corpus_passage:
            continue

        ids = []
        titles = []
        headings = []
        texts = []
        for pid in corpus_passage[k]:
            if pid not in entire_corpus_passage:
                click.echo(f"Error: pid {pid} not found in entire corpus", err=True)
                continue
            ids.append(pid)
            titles.append(entire_corpus_passage[pid]["title"])
            headings.append(entire_corpus_passage[pid]["headings"])
            texts.append(entire_corpus_passage[pid]["text"])

        # Add documents to dataset corpus
        datasets_dict = {}
        datasets_dict["_id"] = ids
        datasets_dict["title"] = titles
        datasets_dict["headings"] = headings
        datasets_dict["text"] = texts

        # Save dataset corpus to disk
        corpus_path = os.path.join(
            TEMP_FOLDER, upload_id, "passage", "corpus_{suffix}.jsonl"
        )
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

        for i in range(0, len(datasets_dict["_id"]), 1_000_000):
            if len(datasets_dict["_id"]) <= 1_000_000:
                suffix = k
                click.echo(f"Saving corpus to {corpus_path.format(suffix=suffix)}")
            else:
                suffix = f"{k}_{int(i/1_000_000)}"
                click.echo(
                    f"Saving corpus {int(i/1_000_000)} to {corpus_path.format(suffix=suffix)}"
                )

            # Save to JSONL
            with open(corpus_path.format(suffix=suffix), "w") as file:
                for j in range(i, min(i + 1_000_000, len(datasets_dict["_id"]))):
                    file.write(
                        json.dumps(
                            {
                                "_id": datasets_dict["_id"][j],
                                "title": datasets_dict["title"][j],
                                "headings": datasets_dict["headings"][j],
                                "text": datasets_dict["text"][j],
                            }
                        )
                        + "\n"
                    )

    del queries_dict_passage
    del corpus_passage
    del entire_corpus_passage_set
    del entire_corpus_passage

    ################################ Documents ################################

    # Fetch entire corpus of documents
    entire_corpus_document_set = set()
    for k in ["core"] + DATASETS["document"]:
        if k not in corpus_document:
            continue
        for docid in corpus_document[k]:
            assert (
                docid not in entire_corpus_document_set
            ), f"Duplicate docid {docid} found in corpus"
            entire_corpus_document_set.add(docid)

    # Load entire corpus of documents
    entire_corpus_document = _prepare_corpus_document(entire_corpus_document_set)
    click.echo(f"Loaded {len(entire_corpus_document)} documents")

    for k in ["core"] + DATASETS["document"]:
        if k not in corpus_document:
            continue

        ids = []
        titles = []
        headings = []
        texts = []
        for docid in corpus_document[k]:
            if docid not in entire_corpus_document:
                click.echo(f"Error: docid {docid} not found in entire corpus", err=True)
                continue
            ids.append(docid)
            titles.append(entire_corpus_document[docid]["title"])
            headings.append(entire_corpus_document[docid]["headings"])
            texts.append(entire_corpus_document[docid]["text"])

        # Add documents to dataset corpus
        datasets_dict = {}
        datasets_dict["_id"] = ids
        datasets_dict["title"] = titles
        datasets_dict["headings"] = headings
        datasets_dict["text"] = texts

        # Save dataset corpus to disk
        corpus_path = os.path.join(
            TEMP_FOLDER, upload_id, "document", "corpus_{suffix}.jsonl"
        )
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

        for i in range(0, len(datasets_dict["_id"]), 1_000_000):
            if len(datasets_dict["_id"]) <= 1_000_000:
                suffix = k
                click.echo(f"Saving corpus to {corpus_path.format(suffix=suffix)}")
            else:
                suffix = f"{k}_{int(i/1_000_000)}"
                click.echo(
                    f"Saving corpus {int(i/1_000_000)} to {corpus_path.format(suffix=suffix)}"
                )

            # Save to JSONL
            with open(corpus_path.format(suffix=suffix), "w") as file:
                for j in range(i, min(i + 1_000_000, len(datasets_dict["_id"]))):
                    file.write(
                        json.dumps(
                            {
                                "_id": datasets_dict["_id"][j],
                                "title": datasets_dict["title"][j],
                                "headings": datasets_dict["headings"][j],
                                "text": datasets_dict["text"][j],
                            }
                        )
                        + "\n"
                    )

    click.echo("Done")
