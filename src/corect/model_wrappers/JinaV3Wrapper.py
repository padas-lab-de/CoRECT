from typing import List, Union

import torch
from transformers import AutoModel

from corect.model_wrappers import AbstractModelWrapper
from corect.utils import cos_sim


def _construct_document(doc):
    if isinstance(doc, str):
        return doc
    elif "title" in doc:
        return f"{doc['title']} {doc['text'].strip()}"
    else:
        return doc["text"].strip()


class JinaV3Wrapper(AbstractModelWrapper):
    def __init__(
        self,
        pretrained_model_name="jinaai/jina-embeddings-v3",
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.encoder = AutoModel.from_pretrained(
            self.pretrained_model_name, trust_remote_code=True
        )
        self.encoder.cuda()
        self.encoder.eval()

    def mean_pooling(
            self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        if self.encoder is None:
            self._lazy_loading()
        if self.encoder.training:
            num_examples = 1 if isinstance(sentences, str) else len(sentences)
            adapter_mask = torch.full(
                (num_examples,), 0, dtype=torch.int32, device=self.device
            )
            sentences = [
                "Represent the query for retrieving evidence documents: " + sentence for sentence in sentences
            ]
            lora_arguments = (
                {"adapter_mask": adapter_mask}
                if adapter_mask is not None
                else {}
            )
            tokens = self.encoder.roberta.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(self.encoder.device) for k, v in tokens.items()}
            token_embs = self.encoder.roberta.forward(**tokens, **lora_arguments)[0]
            token_embs = token_embs.float()
            embeddings = self.mean_pooling(
                token_embs, tokens["attention_mask"]
            )
            return torch.nn.functional.normalize(embeddings, p=2, dim=0)
        return self.encoder.encode(sentences, *args, task="retrieval.query", **kwargs, convert_to_tensor=True)

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        _sentences = [_construct_document(sentence) for sentence in sentences]
        return self.encoder.encode(
            _sentences, *args, task="retrieval.passage", **kwargs, convert_to_tensor=True,
        )

    def get_instructions(self):
        return [
            self.encoder._task_instructions[x]
            for x in ["retrieval.query", "retrieval.passage"]
        ]

    def similarity(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the similarity between two batches of embeddings.
        """
        return cos_sim(embeddings_1, embeddings_2)

    @property
    def device(self):
        return self.encoder.device

    @staticmethod
    def has_instructions():
        return True

    @property
    def name(self) -> str:
        """
        Return the name of the model.
        """
        return "JinaV3Wrapper"
