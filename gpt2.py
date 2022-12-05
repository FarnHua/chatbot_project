# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01-gpt2-with-value-head.ipynb (unless otherwise specified).

__all__ = ['ValueHead', 'GPT2HeadWithValueModel', 'respond_to_batch']

# Cell

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch

# Cell

class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self):
        super().__init__()
        self.detach_head = False
#         self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
#         if self.summary_type == "attn":
#             raise NotImplementedError

        self.summary = Identity()
#         if hasattr(config, "summary_use_proj") and config.summary_use_proj:
#             if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
        num_classes = 1 # config.num_labels
#             else:
#         num_classes = 768 #config.hidden_size
        self.summary = nn.Linear(1024, num_classes)

        self.activation = Identity()
#         if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
        self.activation = nn.Tanh()

        self.first_dropout = Identity()
#         if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
        self.first_dropout = nn.Dropout(0.2) # config.summary_first_dropout

        self.last_dropout = Identity()
#         if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
        self.last_dropout = nn.Dropout(0.2) # config.summary_last_dropout
 
        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

# Cell

class GPT2HeadWithValueModel(nn.Module):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, model_pth):
        super().__init__()
#         config.num_labels = 1
#         self.transformer = GPT2Model(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = GPT2LMHeadModel.from_pretrained(model_pth)
        self.v_head = ValueHead()

#         self.init_weights()

#     def get_output_embeddings(self):
#         return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):

        lm_logits, past, hidden = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )

        output = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )

        lm_logits = output['logits']
        past = output['past_key_values']
        hidden = output['hidden_states']
        hidden_states = hidden[-1]

#         lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + (past,) + (value,)

        return outputs



# +
# Cell

def respond_to_batch(model, queries, sep, mask, txt_len=40, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]
# -


