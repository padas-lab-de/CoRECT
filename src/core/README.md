# CoRE: Constructing a Dataset to Evaluate the Robustness of Compression Techniques

Scripts for creating the CoRE dataset collection. CoRE is short for "Controlled Retrieval Evaluation" and is a collection of datasets to evaluate the robustness of compression techniques for retrieval models. It is cnstructed from the MS MARCO v2 dataset and applies an intelligent corpus sampling method to sample 10k, 100k, 1M, 10M, and 100M passages from the full passage collection. Similarly, 10k, 100k, 1M, and 10M documents are sampled from the full document collection. This corpus sampling strategy avoids trivial sampled corpora that contain no distracting passages or documents, by pooling TREC DL 2023 runs (for the method, see [this paper](https://doi.org/10.1007/978-3-031-88708-6_29)). The queries are human-annotated, paired with each 10 relevant passages and documents, and 100 distractors. The remaining passages and documents are sampled randomly. The dataset is available on [HuggingFace](https://huggingface.co/datasets/anonymousaccount/core).

## Install Dependencies

There are two ways you can install the dependencies to run the code.

### Using Poetry (recommended)

If you have the [Poetry](https://python-poetry.org/) package manager for Python installed already, you can simply set up everything with:

```console
poetry install
source $(poetry env info --path)/bin/activate
```

After the installation of all dependencies, you will end up in a new shell with a loaded venv. In this shell, you can run the main `core` command. You can exit the shell at any time with `exit`.

```console
core --help
```

To install new dependencies in an existing poetry environment, you can run the following commands with the shell environment being activated:

```console
poetry lock
poetry install
```

### Using Pip (alternative)

You can also create a venv yourself and use `pip` to install dependencies:

```console
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Development

### Run Code Formatting

To run the code formatting, you can use the following command:

```console
isort .
black .
```

The order of the commands is important. `isort` will sort the imports in the files, and `black` will format the code.

## Run Code for Creating CoRE

The evaluation code currently supports two datasets: A transformed version of the MS MARCO v2 dataset, called CoRE, and public BEIR datasets.
In addition to the dataset, the code also loads an embedding model (currently Jina V3 or E5-Multilingual) to evaluate the defined compression techniques.
To start the evaluation, execute the commands

```console
core prepare <upload-id>      # Prepare the inital queries, corpus and qrels files
core create <upload-id>       # Create the collection of CoRE datasets (including the sampling of the corpus and queries)
```

After running the evaluation code, you will find the final queries, corpus and qrels files in the `temp` folder. To upload the dataset to HuggingFace, you can run the following command:

```console
core hf-push <upload-id> <repo-name>
```
