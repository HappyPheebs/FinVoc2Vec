import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from transformers import AutoConfig
from .finvoc2vec_config import FinVoc2VecConfig


class FinVoc2Vec(Wav2Vec2PreTrainedModel):
    
    config_class = FinVoc2VecConfig
    
    def __init__(self, model_config: AutoConfig):
        
        super().__init__(model_config)

        self.num_labels = model_config.num_labels
        self.pooling_mode = model_config.pooling_mode
        self.config = model_config
        
        # unpack class weights according to label2id dict
        class_weights = [None] * len(model_config.label2id)
        for label, idx in model_config.label2id.items():
            class_weights[idx] = model_config.class_weights[label]

        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_fct = CrossEntropyLoss(weight=self.class_weights)

        self.wav2vec2 = Wav2Vec2Model(model_config)

        self.dense = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.dropout = nn.Dropout(model_config.final_dropout)
        self.out_proj = nn.Linear(model_config.hidden_size, model_config.num_labels)

        self.init_weights()


    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()


    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        logits = self.dense(hidden_states)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)
        logits = self.out_proj(logits)

        # calc loss if labels are given
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss':loss,
            'logits':logits,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,
        }