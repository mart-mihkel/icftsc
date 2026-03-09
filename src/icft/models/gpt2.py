import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)

from icft.models import PTModel


class PTGPT2Model(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SequenceClassifierOutput | Seq2SeqModelOutput | CausalLMOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        if self.config.task == "causal-lm":
            labels = self._get_causal_labels(labels)

        out = self.base(
            inputs_embeds=inputs,
            attention_mask=attn,
            labels=labels,
            output_hidden_states=True,
        )

        if self.config.task == "seq-cls":
            return self._post_forward_seq_cls(
                input_ids=input_ids,
                labels=labels,
                last_hidden_state=out.hidden_states[-1],
            )

        return out

    def _post_forward_seq_cls(
        self,
        input_ids: Tensor,
        labels: Tensor,
        last_hidden_state: Tensor,
    ) -> SequenceClassifierOutput:
        prompt_ids = self._get_prompt_ids(input_ids)
        batch_size, seq_len = prompt_ids.shape
        device = input_ids.device

        pad_token_id = self.base.config.pad_token_id
        non_pad_mask = (prompt_ids != pad_token_id).to(device, torch.int32)
        token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        logits = self.base.score(last_hidden_state)
        pooled_logits = logits[
            torch.arange(batch_size, device=device),
            last_non_pad_token,
        ]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            pooled_logits.view(-1, self.config.num_labels),
            labels.view(-1),
        )

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)
