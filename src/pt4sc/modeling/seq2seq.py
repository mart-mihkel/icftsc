from torch import Tensor
from transformers.modeling_outputs import Seq2SeqModelOutput

from pt4sc.modeling.common import PTModel


class PTT5ForSeq2SeqLM(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> Seq2SeqModelOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        return self.base(
            labels=labels,
            attention_mask=attn,
            inputs_embeds=inputs,
            output_hidden_states=True,
        )
