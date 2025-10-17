from functools import partial
from typing import List, Dict

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput

from corect.model_wrappers import AbstractModelWrapper
from corect.model_wrappers.utils import transform_func, move_to_cuda, pool
from corect.utils import cos_sim


class SnowflakeWrapper(AbstractModelWrapper):

    def __init__(
        self,
        pretrained_model_name="Snowflake/snowflake-arctic-embed-m-v2.0",
        batch_size: int = 64,
        pool_type: str = "cls",
        doc_as_query: bool = False,
        truncate_dim: int = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.gpu_count = torch.cuda.device_count()
        self.pool_type = pool_type
        self.doc_as_query = doc_as_query
        self.batch_size = batch_size
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()
        self.truncate_dim = truncate_dim
        self.prefix = "query: " if pretrained_model_name.endswith("v2.0") else ("Represent this sentence for searching "
                                                                                "relevant passages: ")

    def encode_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        input_texts = [
            "{}{}".format(self.prefix, q)
            for q in queries
        ]  # vs query: prompt
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> torch.Tensor:
        if self.doc_as_query:
            return self.encode_queries([d["text"] for d in corpus], **kwargs)

        input_texts = [
            "{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus
        ]
        input_texts = ["{}".format(t) for t in input_texts]  # No doc prefix
        return self._do_encode(input_texts)

    def _do_encode(self, input_texts: List[str]) -> torch.Tensor:
        dataset: Dataset = Dataset.from_dict({"contents": input_texts})
        dataset.set_transform(partial(transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True,
        )

        encoded_embeds = []
        for batch_dict in data_loader:
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                    self.pool_type,
                )
                encoded_embeds.append(embeds)

        embeds = torch.vstack(encoded_embeds)
        return torch.nn.functional.normalize(embeds, p=2, dim=1)

    def similarity(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the similarity between two batches of embeddings.
        """
        return cos_sim(embeddings_1, embeddings_2)

    @property
    def name(self) -> str:
        """
        Return the name of the model.
        """
        suffix = (
            f"-truncated{self.truncate_dim}" if self.truncate_dim is not None else ""
        )
        return "SnowflakeWrapper"
