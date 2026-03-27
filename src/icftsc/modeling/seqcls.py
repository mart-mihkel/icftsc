import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
)
from transformers.utils.generic import ModelOutput

from icftsc.modeling.pt import PTModel


class PTBertForSequenceClassification(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> ModelOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        return self.base(inputs_embeds=inputs, attention_mask=attn, labels=labels)


class PTGPTForSequenceClassification(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> SequenceClassifierOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        prompt_ids = self._get_prompt_ids(input_ids)
        batch_size, seq_len = prompt_ids.shape
        device = input_ids.device

        base_out = self.base(
            labels=labels,
            attention_mask=attn,
            inputs_embeds=inputs,
            output_hidden_states=True,
        )

        last_hidden_state = base_out.hidden_states[-1]
        pad_token_id = self.base.config.pad_token_id
        non_pad_mask = (prompt_ids != pad_token_id).to(device, torch.int32)
        token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
        batch_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        logits = self.base.score(last_hidden_state)
        pooled_logits = logits[batch_indices, last_non_pad_token]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                pooled_logits.view(-1, self.config.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)


class PTT5ForSequenceClassification(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> Seq2SeqSequenceClassifierOutput:
        pad_token_id = self.base.config.pad_token_id
        prompt_ids = self._get_prompt_ids(input_ids)
        batch_size, seq_len = prompt_ids.shape
        device = input_ids.device

        enc_inputs, enc_attn = self._get_prompt(input_ids, attention_mask)

        dec_input_ids = self._shift_inputs(input_ids)
        dec_attention_mask = self._shift_attention(attention_mask)
        dec_inputs, dec_attn = self._get_prompt(dec_input_ids, dec_attention_mask)

        out = self.base.transformer(
            inputs_embeds=enc_inputs,
            attention_mask=enc_attn,
            decoder_inputs_embeds=dec_inputs,
            decoder_attention_mask=dec_attn,
            labels=labels,
        )

        last_hidden_state = out.last_hidden_state
        non_pad_mask = (prompt_ids != pad_token_id).to(device, torch.int32)
        token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
        batch_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        logits = self.base.classification_head(last_hidden_state)
        pooled_logits = logits[batch_indices, last_non_pad_token]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                pooled_logits.view(-1, self.config.num_labels),
                labels.view(-1),
            )

        return Seq2SeqSequenceClassifierOutput(loss=loss, logits=pooled_logits)
