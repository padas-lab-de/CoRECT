import os

import click
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from core.cli.huggingface.readme import readme_template
from core.config import *
from core.utils import *

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


@click.command()
@click.argument("upload_id", type=str)
@click.argument("repo_id", type=str)
def hf_push(upload_id: str, repo_id: str):
    """
    Pushing the collection of datasets to HuggingFace.
    """
    # Login to HuggingFace
    load_dotenv()
    login(token=os.getenv("DEFAULT_HF_TOKEN"))

    # Create the repository if it does not exist
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload qrels and queries
    qrels_passage_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "passage-final.jsonl"
    )
    qrels_document_path = os.path.join(
        TEMP_FOLDER, upload_id, "qrels", "document-final.jsonl"
    )
    queries_path = os.path.join(TEMP_FOLDER, upload_id, "queries-final.jsonl")

    # Create README information
    format_dataset_info = ""
    format_configs = ""

    # Fill with values from qrels and queries
    format_dataset_info += f"  - config_name: qrels\n"
    format_dataset_info += f"    features:\n"
    format_dataset_info += f"      - name: query-id\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"      - name: corpus-id\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"      - name: type\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"    splits:\n"
    format_dataset_info += f"      - name: passage\n"
    format_dataset_info += f"        num_bytes: {os.path.getsize(qrels_passage_path)}\n"
    format_dataset_info += f"        num_examples: {count_lines(qrels_passage_path)}\n"
    format_dataset_info += f"      - name: document\n"
    format_dataset_info += (
        f"        num_bytes: {os.path.getsize(qrels_document_path)}\n"
    )
    format_dataset_info += f"        num_examples: {count_lines(qrels_document_path)}\n"
    format_dataset_info += f"  - config_name: queries\n"
    format_dataset_info += f"    features:\n"
    format_dataset_info += f"      - name: _id\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"      - name: text\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"    splits:\n"
    format_dataset_info += f"      - name: test\n"
    format_dataset_info += f"        num_bytes: {os.path.getsize(queries_path)}\n"
    format_dataset_info += f"        num_examples: {count_lines(queries_path)}\n"
    format_dataset_info += f"  - config_name: corpus\n"
    format_dataset_info += f"    features:\n"
    format_dataset_info += f"      - name: _id\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"      - name: title\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"      - name: headings\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"      - name: text\n"
    format_dataset_info += f"        dtype: string\n"
    format_dataset_info += f"    splits:\n"
    format_configs += f"  - config_name: qrels\n"
    format_configs += f"    data_files:\n"
    format_configs += f"      - split: passage\n"
    format_configs += f"        path: qrels/passage.jsonl\n"
    format_configs += f"      - split: document\n"
    format_configs += f"        path: qrels/document.jsonl\n"
    format_configs += f"  - config_name: queries\n"
    format_configs += f"    data_files:\n"
    format_configs += f"      - split: test\n"
    format_configs += f"        path: queries.jsonl\n"
    format_configs += f"  - config_name: corpus\n"
    format_configs += f"    data_files:\n"

    # Upload files to the repository
    api.upload_file(
        path_or_fileobj=qrels_passage_path,
        path_in_repo="qrels/passage.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=qrels_document_path,
        path_in_repo="qrels/document.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=queries_path,
        path_in_repo="queries.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Upload corpora to the repository
    for document_length in DATASETS:
        click.echo(f"Document length: {document_length}")

        for k in ["core"] + DATASETS[document_length]:

            if document_length == "passage":
                split_name = "pass"
            elif document_length == "document":
                split_name = "doc"
            else:
                raise ValueError(f"Unknown document length: {document_length}")

            if k == "core":
                split_name += "_core"
            elif k == 10_000:
                split_name += "_10k"
            elif k == 100_000:
                split_name += "_100k"
            elif k == 1_000_000:
                split_name += "_1M"
            elif k == 10_000_000:
                split_name += "_10M"
            elif k == 100_000_000:
                split_name += "_100M"
            else:
                raise ValueError(f"Unknown k: {k}")

            if k == "core" or k <= 1_000_000:
                corpus_path = os.path.join(
                    TEMP_FOLDER, upload_id, document_length, f"corpus_{k}.jsonl"
                )
                assert os.path.exists(
                    corpus_path
                ), f"Corpus file {corpus_path} does not exist"

                num_bytes = os.path.getsize(corpus_path)
                num_examples = count_lines(corpus_path)

                format_dataset_info += f"      - name: {split_name}\n"
                format_dataset_info += f"        num_bytes: {num_bytes}\n"
                format_dataset_info += f"        num_examples: {num_examples}\n"
                format_configs += f"      - split: {split_name}\n"
                format_configs += f"        path: {document_length}/corpus_{k}.jsonl\n"

                # Upload corpus.jsonl to the repository
                api.upload_file(
                    path_or_fileobj=corpus_path,
                    path_in_repo=f"{document_length}/corpus_{k}.jsonl",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            else:
                num_bytes = 0
                num_examples = 0

                # For larger datasets, we need to upload multiple files
                i = 0
                while True:
                    corpus_path = os.path.join(
                        TEMP_FOLDER, upload_id, document_length, f"corpus_{k}_{i}.jsonl"
                    )

                    if not os.path.exists(corpus_path):
                        break

                    num_bytes += os.path.getsize(corpus_path)
                    num_examples += count_lines(corpus_path)

                    # Upload corpus.jsonl to the repository
                    api.upload_file(
                        path_or_fileobj=corpus_path,
                        path_in_repo=f"{document_length}/corpus_{k}_{i}.jsonl",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

                    # Increment counter
                    i += 1

                format_dataset_info += f"      - name: {split_name}\n"
                format_dataset_info += f"        num_bytes: {num_bytes}\n"
                format_dataset_info += f"        num_examples: {num_examples}\n"
                format_configs += f"      - split: {split_name}\n"
                format_configs += (
                    f"        path: {document_length}/corpus_{k}_*.jsonl\n"
                )

    # Remove trailing newlines
    format_dataset_info = format_dataset_info.strip("\n")
    format_configs = format_configs.strip("\n")

    # Create README.md
    readme_path = os.path.join(TEMP_FOLDER, upload_id, "README.md")
    readme_text = readme_template.format(
        format_dataset_info,
        format_configs,
    )
    with open(readme_path, "w") as f:
        f.write(readme_text)

    # Upload README.md to the repository
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
