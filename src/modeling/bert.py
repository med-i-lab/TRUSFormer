from dataclasses import dataclass


@dataclass
class TransformerConfig:

    _target_: str = __name__ + ".TransformerEncoder"

    hidden_size: int = 64
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    intermediate_size: int = 64
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1


def TransformerEncoder(
    hidden_size: int = 256,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 4,
    intermediate_size: int = 512,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
):
    from transformers.models.bert.configuration_bert import BertConfig
    from transformers.models.bert.modeling_bert import BertForSequenceClassification

    cfg = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
    )

    model = BertForSequenceClassification(cfg).bert.encoder

    return model
