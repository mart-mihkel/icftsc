from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput

from pt4sc.modeling.common import PTModel


class PTGPTForCausalLM(PTModel):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        if labels is not None:
            labels = self._get_causal_labels(labels)

        return self.base(
            labels=labels,
            attention_mask=attn,
            inputs_embeds=inputs,
            output_hidden_states=True,
        )
