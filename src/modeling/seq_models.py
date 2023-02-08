from dataclasses import dataclass
from torch.nn import Module
import torch
from .positional_embedding import GridPositionEmbedder2d
from .pca import TorchPCA


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


class ExactSeqModel(torch.nn.Module):
    def __init__(
        self,
        pool_mode="mean",
        in_feats=512,
        feature_reduction=None,
        hidden_size=512,
        num_layers=4,
        num_attn_heads=4,
        intermediate_size=512,
        patch_dropout=0.1,
        inner_dropout=0.1,
        use_pos_embeddings=True,
        grid_shape=(28, 46),
    ):
        super().__init__()

        self.transformer = TransformerEncoder(
            hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=inner_dropout,
            attention_probs_dropout_prob=inner_dropout,
        )

        if feature_reduction == "pca":
            self.feature_reduction = TorchPCA(intermediate_size)
            self.fit_latent_space = self.feature_reduction.fit
        if feature_reduction == "linear":
            self.feature_reduction = torch.nn.Linear(in_feats, out_features=hidden_size)
        if feature_reduction is None:
            self.feature_reduction = torch.nn.Identity()

        self.pos_embedding = GridPositionEmbedder2d(hidden_size, grid_shape)
        self.use_pos_embeddings = use_pos_embeddings

        if pool_mode == "cls":
            self.cls_token = torch.nn.Parameter(torch.randn((1, hidden_size)))

        self.pool_mode = pool_mode
        self.patch_dropout = patch_dropout
        self.features_dim = hidden_size

    def forward(self, X, pos_x, pos_y, output_attentions=False):

        X = self.feature_reduction(X)

        if self.use_pos_embeddings:
            pos_emb = self.pos_embedding(pos_x, pos_y)
            X = X + pos_emb

        if self.patch_dropout and self.training:
            num_patches = len(X)
            perm = torch.randperm(X.size(0))
            k = int((1 - self.patch_dropout) * num_patches)
            idx = perm[:k]
            X = X[idx]

        if self.pool_mode == "cls":
            X = torch.concat([self.cls_token, X], dim=0)

        out = self.transformer(X.unsqueeze(0), output_attentions=output_attentions)
        X = out.last_hidden_state[0]

        X = X[0] if self.pool_mode == "cls" else X.mean(dim=0)

        if output_attentions:
            return {"pool_output": X, "attentions": out.attentions}

        return X


class ExactSeqClassifier(Module):
    def __init__(self, feat_extractor, seq_model, linear_layer):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.seq_model = seq_model
        self.linear_layer = linear_layer

    def forward(self, patches, X, Y):
        n_patches, n_chans, h, w = patches.shape

        features = self.feat_extractor(patches)
        out = self.seq_model(features, X, Y, output_attentions=True)
        out["logits"] = self.linear_layer(out["pool_output"])
        return out


def core_classifier_batch_to_output(model: ExactSeqClassifier, batch, device):
    patch, pos, label, metadata = batch
    patch = patch[0].to(device)
    x = pos[0, :, 0].to(device)
    y = pos[0, :, 2].to(device)

    out = model(patch, x, y)
    logits = out["logits"]

    classification_output = {
        "pool_output": out["pool_output"].unsqueeze(0),
        "logits": logits.unsqueeze(0),
        "preds": logits.unsqueeze(0).softmax(-1),
        "labels": label.long(),
        **metadata,
    }

    attentions = out["attentions"]

    return {"classification_output": classification_output, "attentions": attentions}
