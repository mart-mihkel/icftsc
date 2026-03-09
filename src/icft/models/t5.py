from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)

from icft.models import PTModel


class PTT5Model(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SequenceClassifierOutput | Seq2SeqModelOutput | CausalLMOutput:
        enc_inputs, enc_attn = self._get_prompt(input_ids, attention_mask)

        dec_input_ids = self._shift_inputs(input_ids)
        dec_attention_mask = self._shift_attention(attention_mask)
        dec_inputs, dec_attn = self._get_prompt(dec_input_ids, dec_attention_mask)

        if self.config.task == "seq-cls":
            out = self.base.transformer(
                inputs_embeds=enc_inputs,
                attention_mask=enc_attn,
                decoder_inputs_embeds=dec_inputs,
                decoder_attention_mask=dec_attn,
                labels=labels,
            )

            return self._post_forward_seq_cls(
                input_ids=input_ids,
                labels=labels,
                last_hidden_state=out.last_hidden_state,
            )

        return self.base(
            inputs_embeds=enc_inputs,
            attention_mask=enc_attn,
            labels=labels,
        )

    def _shift_inputs(self, input_ids: Tensor) -> Tensor:
        decoder_start_token_id = self.base.config.decoder_start_token_id
        pad_token_id = self.base.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("No decoder start token id")

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("No pad token id")

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def _shift_attention(self, attention_mask: Tensor) -> Tensor:
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(
            max=attention_mask.size(1) - 1
        )

        return attention_mask.clone().scatter_(1, lengths, 1)

    def _post_forward_seq_cls(
        self,
        input_ids: Tensor,
        labels: Tensor,
        last_hidden_state: Tensor,
    ) -> SequenceClassifierOutput:
        prompt_ids = self._get_prompt_ids(input_ids)
        batch_size, seq_len = prompt_ids.shape

        logits = self.base.classification_head(last_hidden_state)
        pooled_logits = logits[:, 0]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            pooled_logits.view(-1, self.config.num_labels),
            labels.view(-1),
        )

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)
