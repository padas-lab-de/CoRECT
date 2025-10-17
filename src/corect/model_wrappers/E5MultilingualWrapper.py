from typing import List, Dict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from corect.model_wrappers import AbstractModelWrapper
from corect.model_wrappers.utils import create_batch_dict, move_to_cuda
from corect.utils import cos_sim


class E5MultilingualWrapper(AbstractModelWrapper):
    def __init__(
        self,
        pretrained_model_name="intfloat/multilingual-e5-large-instruct",
        truncate_dim: int = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.encoder.cuda()
        self.encoder.eval()
        self.query_instruct = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        self.truncate_dim = truncate_dim

    @staticmethod
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self.query_instruct}\nQuery: {query}"

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.get_detailed_instruct(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = [
            "{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus
        ]
        return self._do_encode(input_texts)

    def _do_encode(self, input_texts: List[str]) -> torch.Tensor:
        encoded_embeds = []
        batch_size = 64
        for start_idx in range(0, len(input_texts), batch_size):
            batch_input_texts: List[str] = input_texts[
                start_idx : start_idx + batch_size
            ]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast("cuda"):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = self.average_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds)
        if self.truncate_dim is not None:
            encoded_embeds = [e[:, : self.truncate_dim] for e in encoded_embeds]

        return torch.vstack(encoded_embeds)

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
        return f"E5MultilingualWrapper{suffix}"
