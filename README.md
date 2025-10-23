# CoRECT: A Framework for Evaluating Embedding Compression Techniques at Scale

Experiments on the robustness of embedding compression methods comparing scalar and binary quantization, floating point casting, vector truncation, principal component analysis, locality sensitive hashing and product quantization.

## Install Dependencies

There are two ways to install the dependencies and run the code.

### Using Poetry (recommended)

If you have the [Poetry](https://python-poetry.org/) package manager for Python installed already, you can simply set up everything with:

```console
poetry install
source $(poetry env info --path)/bin/activate
```

After the installation of all dependencies, you will end up in a new shell with a loaded venv. In this shell, you can run the main `corect` command. You can exit the shell at any time with `exit`.

```console
corect --help
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

## Run Evaluation Code

The evaluation code currently supports two datasets: A transformed version of the MS MARCO v2 dataset, called CoRE, and public BeIR datasets.
In addition to the dataset, the code also loads an embedding model to evaluate the defined compression techniques.
The currently supported models are [Jina V3](https://huggingface.co/jinaai/jina-embeddings-v3) (jinav3), [Multilingual-E5-Large-Instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) (e5), [Snowflake-Arctic-Embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) (snowflake) and [Snowflake-Arctic-Embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) (snowflakev2).
To start the evaluation, execute the command

```console
corect evaluate jinav3 core     # Evaluates Jina V3 on CoRE
corect evaluate e5 beir         # Evaluates E5-Multilingual on BeIR
```

The code downloads the respective datasets from Hugging Face and uses the chosen model to generate the embeddings.
By default, the embeddings are stored on device for later re-evaluation.
To avoid this, change line *177* in the `evaluation.py` script to `None`.
The embeddings will then be compressed using the compression methods specified in `compression_registry.py`.

After running the evaluation code, you will find the results in the `results` folder.
The results are stored in a JSON file in a folder structure organized by model name and dataset.
To share the results, copy the respective JSON file to the `share_results` folder.
Default folders for storing results and embeddings can be changed in `config.py`.
Results are stored in the following format:

```json
{
    "ndcg_at_1": 0.38462,
    "ndcg_at_3": 0.33752,
    "ndcg_at_5": 0.30636,
    "ndcg_at_10": 0.24977,
    "ndcg_at_20": 0.31123,
    "ndcg_at_100": 0.51075,
    "ndcg_at_200": 0.55959,
    "ndcg_at_300": 0.56132,
    "ndcg_at_500": 0.56132,
    "ndcg_at_1000": 0.56132,
    "map_at_1": 0.03846,
    "map_at_3": 0.08077,
    "map_at_5": 0.10708,
    "map_at_10": 0.1392,
    "map_at_20": 0.17026,
    "map_at_100": 0.23058,
    "map_at_200": 0.24235,
    "map_at_300": 0.24262,
    "map_at_500": 0.24262,
    "map_at_1000": 0.24262,
    "recall_at_1": 0.03846,
    "recall_at_3": 0.09692,
    "recall_at_5": 0.14154,
    "recall_at_10": 0.21385,
    "recall_at_20": 0.32462,
    "recall_at_100": 0.84308,
    "recall_at_200": 0.99385,
    "recall_at_300": 1.0,
    "recall_at_500": 1.0,
    "recall_at_1000": 1.0,
    "precision_at_1": 0.38462,
    "precision_at_3": 0.32308,
    "precision_at_5": 0.28308,
    "precision_at_10": 0.21385,
    "precision_at_20": 0.16231,
    "precision_at_100": 0.08431,
    "precision_at_200": 0.04969,
    "precision_at_300": 0.03333,
    "precision_at_500": 0.02,
    "precision_at_1000": 0.01,
    "mrr_at_1": 0.38462,
    "mrr_at_3": 0.48462,
    "mrr_at_5": 0.50385,
    "mrr_at_10": 0.51581,
    "mrr_at_20": 0.52769,
    "mrr_at_100": 0.52923,
    "mrr_at_200": 0.52923,
    "mrr_at_300": 0.52923,
    "mrr_at_500": 0.52923,
    "mrr_at_1000": 0.52923,
    "rc_at_1": {
        "relevant": 0.38462,
        "distractor": 0.61538
    },
    "rc_at_3": {
        "relevant": 0.96923,
        "distractor": 2.03077
    },
    "rc_at_5": {
        "relevant": 1.41538,
        "distractor": 3.58462
    },
    "rc_at_10": {
        "relevant": 2.13846,
        "distractor": 7.83077
    },
    "rc_at_20": {
        "relevant": 3.24615,
        "distractor": 16.69231
    },
    "rc_at_100": {
        "relevant": 8.43077,
        "distractor": 88.69231
    },
    "rc_at_200": {
        "relevant": 9.93846,
        "distractor": 98.10769
    },
    "rc_at_300": {
        "relevant": 10.0,
        "distractor": 98.83077
    },
    "rc_at_500": {
        "relevant": 10.0,
        "distractor": 99.36923
    },
    "rc_at_1000": {
        "relevant": 10.0,
        "distractor": 99.63077
    },
    "rankings": {
        "qid1": {
            "relevant": {
                "cid1": 0,
                "cid9": 5,
                ...
            },
            "distractor": {
                "cid3": 2,
                "cid5": 3,
                ...
            },
            "random": {
                "cid17": 1,
                "cid15": 11,
                ...
            }
        },
        "qid2": {
            ...
        },
        ...
    }
}
```

## Extend CoRECT

### Add New Compression Technique

The currently implemented compression techniques can be found in the [quantization](src/corect/quantization) folder.
To add a new method, implement a class that extends [AbstractCompression](src/corect/quantization/AbstractCompression.py) and add your custom compression technique via the `compress()` method.

```python
import torch

from corect.quantization.AbstractCompression import AbstractCompression


PRECISION_TYPE = {
    "float16": 16,
    "bfloat16": 16,
}


class FloatingCompression(AbstractCompression):

    def __init__(self, precision_type: str = "float16"):
        assert precision_type in PRECISION_TYPE
        self.precision_type = precision_type

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.precision_type == "float16":
            return embeddings.type(torch.float16)
        elif self.precision_type == "bfloat16":
            return embeddings.type(torch.bfloat16)
        else:
            raise NotImplementedError(
                f"Cannot convert embedding to invalid precision type {self.precision_type}!"
            )
```

To include your class in the evaluation, modify the `add_compressions` method in the [compression registry](src/corect/compression_registry.py) to register your class with the compression methods dictionary.
Adding the previously defined `FloatingCompression` class looks like this:

````python
from typing import Dict

from corect.quantization.AbstractCompression import AbstractCompression
from corect.quantization.FloatCompression import PRECISION_TYPE, FloatCompression


class CompressionRegistry:

    _compression_methods: Dict[str, AbstractCompression] = {}

    @classmethod
    def get_compression_methods(cls) -> Dict[str, AbstractCompression]:
        return cls._compression_methods

    @classmethod
    def clear(cls):
        cls._compression_methods.clear()

    @classmethod
    def add_baseline(cls):
        cls._compression_methods["32_full"] = FloatCompression("full")

    @classmethod
    def add_compressions(cls):
        # Add your compression method here to use it for evaluation.
        for precision, num_bits in PRECISION_TYPE.items():
            cls._compression_methods[f"{num_bits}_{precision}"] = FloatCompression(
                precision
            )
````

You should now be able to evaluate your compression technique by running the evaluation script as described above.

### Add New Model

New embedding models can be added by implementing the [AbstractModelWrapper](src/corect/model_wrappers/AbstractModelWrapper.py) class, which requires implementing encoding functions for queries and documents.
Any model available via `transformers` can be added easily.
For reference, consider the example below:

```python
from typing import List, Union

import torch
from transformers import AutoModel, AutoTokenizer

from corect.model_wrappers import AbstractModelWrapper
from corect.utils import cos_sim


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class Qwen3Wrapper(AbstractModelWrapper):
    
    def __init__(self, pretrained_model_name="Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True,
                                                 torch_dtype=torch.float16)
        self.encoder.cuda()
        self.encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True,
                                                       padding_side='left')

    def _encode_input(self, sentences: List[str]) -> torch.tensor:
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        inputs.to('cuda')
        model_outputs = self.encoder(**inputs)
        outputs = _last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return outputs

    def encode_queries(self, queries: List[str], **kwargs) -> torch.tensor:
        return self._encode_input(queries)

    def encode_corpus(self, corpus: Union[str, List[str]], **kwargs) -> torch.tensor:
        if isinstance(corpus, str):
            corpus = [corpus]
        return self._encode_input(corpus)

    def similarity(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> torch.Tensor:
        return cos_sim(embeddings_1, embeddings_2)


    @property
    def name(self) -> str:
        return "Qwen3Wrapper"
```

The wrapper then needs to be registered in the `get_model_wrapper()` method of the [evaluation script](src/corect/cli/evaluate.py).

```python
from typing import Tuple

from corect.model_wrappers import  AbstractModelWrapper, JinaV3Wrapper, Qwen3Wrapper


def get_model_wrapper(model_name: str) -> Tuple[AbstractModelWrapper, int]:
    if model_name == "jinav3":
        return JinaV3Wrapper(), 1024
    elif model_name == "qwen3":
        return Qwen3Wrapper(), 1024     # 1024 is the embedding dimension of the Qwen model.
    else:
        raise NotImplementedError(f"Model {model_name} not supported!")
```

The model can then be evaluated as follows:

```
corect evaluate qwen3 core
```

### Add New Dataset

Our framework supports the addition of any HuggingFace retrieval datasets with corpus, queries and qrels splits.
To add a custom dataset, navigate to the [dataset utils](src/corect/dataset_utils.py) script, add a load function for your new dataset and register it in the `load_data()` function.
You also need to add information on the new dataset to the `datasets` dictionary in this class in the form of `datasets[<dataset_name>]=[<dataset_name>]`.
The example below adds a new dataset called `my_ir_dataset`:

```python
from collections import defaultdict
from typing import Dict, Tuple

from datasets import load_dataset

CoRE = {
    "passage": {
        "pass_core": 10_000,
        "pass_10k": 10_000,
    },
}
DATASET = [
    "my_dataset_name"
]
CoRE_NAME = "core"
DATASET_NAME = "my_ir_dataset"
DATASETS = {CoRE_NAME: CoRE, DATASET_NAME: DATASET}


def _load_core_data(dataset_sub_corpus: str):
    # Code for loading CoRE
    ...


def _load_my_dataset(
        dataset_name: str
) -> Tuple[defaultdict, Dict[str, str], defaultdict, defaultdict]:
    dataset_queries = load_dataset(f"hf_repo/my_dataset", "queries")
    dataset_qrels = load_dataset(f"hf_repo/my_dataset", "default")
    dataset_corpus = load_dataset(f"hf_repo/my_dataset", "corpus")
    
    # Transform dataset
    qrels = defaultdict(dict)
    for q in dataset_qrels:
        query_id = q["query-id"]
        corpus_id = q["corpus-id"]
        qrels[query_id][corpus_id] = int(q["score"])

    queries = {q["_id"]: q["text"] for q in dataset_queries["queries"] if q["_id"] in qrels.keys()}

    corpora = defaultdict(dict)
    for d in dataset_corpus["corpus"]:
        corpora[dataset_name][d["_id"]] = {
            "title": d["title"],
            "text": d["text"],
        }

    return corpora, queries, qrels, qrels


def load_data(dataset_name: str, dataset_sub_corpus: str):
    if dataset_name == CoRE_NAME:
        return _load_core_data(dataset_sub_corpus)
    elif dataset_name == DATASET_NAME:
        return _load_my_dataset(dataset_sub_corpus)
    else:
        raise NotImplementedError(
            f"Cannot load data for unsupported dataset {dataset_name}!"
        )
```

Running the evaluation script on the new dataset can then be achieved by executing the evaluation command:

```
corect evaluate jinav3 my_ir_dataset
```

## Citation

If you use this project in your research, please cite the following paper:

```
@misc{caspari2025corect,
      title={CoRECT: A Framework for Evaluating Embedding Compression Techniques at Scale}, 
      author={L. Caspari and M. Dinzinger and K. Gosh Dastidar and C. Fellicious and J. Mitrović and M. Granitzer},
      year={2025},
      eprint={2510.19340},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2510.19340}, 
}
```
