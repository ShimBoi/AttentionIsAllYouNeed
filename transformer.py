import torch
import torch.nn as nn
from positional_encoders import sinusoidal_positional_encoding
from utils import TransformerConfig


class SDPA(nn.Module):
    """
    Scaled Dot-Product Attention:
    """

    def forward(self, Q, K, V, mask=None):
        """
        Q: [..., seqlen_q, d_k]
        K: [..., seqlen_k, d_k]
        V: [..., seqlen_v, d_v]
        """
        d_k = Q.shape[-1]

        scores = Q @ K.transpose(-2, -1)  # [..., seqlen_q, seqlen_k]
        scaled_scores = scores / torch.sqrt(
            torch.tensor(d_k, dtype=Q.dtype, device=Q.device)
        )  # scaled for softmax vanishing gradient

        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, -float("inf"))

        attention = torch.softmax(scaled_scores, dim=-1)
        output = attention @ V  # [..., seqlen_q, d_v]
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention:
    """

    def __init__(self, d_model: int, num_heads: int, d_k: int = None, d_v: int = None):
        """
        Q: [..., seqlen_q, d_k]
        K: [..., seqlen_k, d_k]
        V: [..., seqlen_v, d_v]
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_k or d_model // num_heads
        self.d_v = d_v or d_model // num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.sdpa = SDPA()
        self.W_o = nn.Linear(num_heads * self.d_v, d_model)

    def forward(self, query, key, value, mask=None):
        """
        query: [bs, seqlen_q, d_model]
        key: [bs, seqlen_k, d_model]
        value: [bs, seqlen_v, d_model]
        """
        bs = query.shape[0]

        # split vectors for each attention head
        # [bs, seqlen, d_model] --> [bs, num_heads, seqlen, d_{k,v}]
        Q = self.W_q(query).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(bs, -1, self.num_heads, self.d_v).transpose(1, 2)

        attention = self.sdpa(
            Q=Q, K=K, V=V, mask=mask
        )  # [bs, num_heads, seqlen_q, d_v]
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(bs, -1, self.num_heads * self.d_v)
        )  # [bs, seqlen_q, num_heads * d_v]

        return self.W_o(attention)


class FFN(nn.Module):
    """
    Implementation of position-wise fully connected feed-forward network
    """

    def __init__(self, d_model, ff_d):
        super().__init__()
        self.expand = nn.Linear(d_model, ff_d)
        self.contract = nn.Linear(ff_d, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.expand(x)
        x = self.relu(x)
        output = self.contract(x)
        return output


class EncoderBlock(nn.Module):
    """
    Implementation of the encoder block of transformer
    """

    def __init__(
        self,
        ff_d: int,
        d_model: int,
        num_heads: int,
        d_k: int = None,
        d_v: int = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model=d_model, ff_d=ff_d)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, input_embedding, padding_mask=None):
        residual = input_embedding
        x = self.mha(
            query=input_embedding,
            key=input_embedding,
            value=input_embedding,
            mask=padding_mask,
        )
        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)

        residual = x
        x = self.ff(x)
        x = self.dropout2(x)
        output = self.layer_norm2(residual + x)
        return output


class DecoderBlock(nn.Module):
    """
    Implementation of the decoder block of transformer
    """

    def __init__(
        self,
        ff_d: int,
        d_model: int,
        num_heads: int,
        d_k: int = None,
        d_v: int = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.masked_mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v
        )
        self.mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.ff = FFN(d_model=d_model, ff_d=ff_d)

    def forward(
        self,
        output_embedding,
        encoder_output,
        self_attn_mask=None,
        cross_attn_mask=None,
    ):
        residual = output_embedding
        x = self.masked_mha(
            query=output_embedding,
            key=output_embedding,
            value=output_embedding,
            mask=self_attn_mask,
        )
        x = self.dropout1(x)
        x = self.layer_norm1(residual + x)

        residual = x
        x = self.mha(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=cross_attn_mask,
        )
        x = self.dropout2(x)
        x = self.layer_norm2(residual + x)

        residual = x
        x = self.ff(x)
        x = self.dropout3(x)
        output = self.layer_norm3(residual + x)
        return output


class Transformer(nn.Module):
    """
    Complete implementation of the encoder-decoder transformer from the original "Attention Is All You Need Paper"
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.device = cfg.device

        self.n_encoder_layers = cfg.n_encoder_layers
        self.n_decoder_layers = cfg.n_decoder_layers

        self.d_model = cfg.d_model
        self.d_ff = cfg.d_ff
        self.num_heads = cfg.num_heads
        self.d_k = cfg.d_k
        self.d_v = cfg.d_v
        self.vocab_size = cfg.vocab_size
        self.dropout_prob = cfg.dropout_prob

        match (cfg.positional_encoding):
            case "sinusoidal":
                self.positional_encoder = sinusoidal_positional_encoding
            case _:
                raise NotImplementedError(
                    "only sinusoidal positional encodings supported"
                )

        # layers
        self.embedding = nn.Embedding(
            self.vocab_size, self.d_model
        )  # shared for both encoder and decoder
        self.scale = torch.sqrt(torch.tensor(self.d_model, device=self.device))
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.dropout2 = nn.Dropout(self.dropout_prob)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    ff_d=self.d_ff,
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_k=self.d_k,
                    d_v=self.d_v,
                )
                for _ in range(self.n_encoder_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    ff_d=self.d_ff,
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_k=self.d_k,
                    d_v=self.d_v,
                )
                for _ in range(self.n_decoder_layers)
            ]
        )
        self.output_proj = nn.Linear(self.d_model, self.vocab_size)
        self.output_proj.weight = (
            self.embedding.weight
        )  # weight also shared with linear projection
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, input, padding_mask=None):
        embed = self.embedding(input)
        embed *= self.scale
        x = self.positional_encoder(embed)
        x = self.dropout1(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(input_embedding=x, padding_mask=padding_mask)

        return x

    def decode(self, input, encoder_output, cross_attn_mask=None, self_attn_mask=None):
        embed = self.embedding(input)
        embed *= self.scale
        x = self.positional_encoder(embed)
        x = self.dropout2(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(
                output_embedding=x,
                encoder_output=encoder_output,
                cross_attn_mask=cross_attn_mask,
                self_attn_mask=self_attn_mask,
            )

        logits = self.output_proj(x)
        # probs = self.softmax(x)
        return logits

    def forward(self, input, target, input_padding_mask=None, self_attn_mask=None):
        encoder_output = self.encode(input, padding_mask=input_padding_mask)
        logits = self.decode(
            target,
            encoder_output,
            cross_attn_mask=input_padding_mask,
            self_attn_mask=self_attn_mask,
        )
        return logits
