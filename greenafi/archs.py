"""Model builders for the energy study.

Compute cost (FLOPs, latency, memory, energy) depends on the architecture and the
input shape, not on the numerical values of the weights. So every builder can
return either the real pretrained checkpoint (``pretrained=True``, needs
HuggingFace access) or a randomly initialised model with the *exact* same
architecture (``pretrained=False``). Anything that depends on the weights
(accuracy, embedding fidelity after quantisation) must use ``pretrained=True``.
"""
from __future__ import annotations

import copy
import os

import torch
from torch import nn
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)

XLSR_ID = "facebook/wav2vec2-large-xlsr-53"
WHISPER_ID = "openai/whisper-medium"

# Architecture of facebook/wav2vec2-large-xlsr-53 (config.json on the Hub).
XLSR53_CONFIG = dict(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_act="gelu",
    feat_extract_norm="layer",
    feat_extract_activation="gelu",
    conv_dim=(512, 512, 512, 512, 512, 512, 512),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_bias=True,
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
    do_stable_layer_norm=True,
    layerdrop=0.0,
    mask_time_prob=0.0,
)

# Architecture of openai/whisper-medium (config.json on the Hub).
WHISPER_MEDIUM_CONFIG = dict(
    d_model=1024,
    encoder_layers=24,
    decoder_layers=24,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
    num_mel_bins=80,
    vocab_size=51865,
    max_source_positions=1500,
    max_target_positions=448,
)


def bake_weight_norm(model: nn.Module) -> nn.Module:
    """Replace the weight-norm parametrisation of the positional convolution by
    its computed weight. Outputs are unchanged (the weight is the same tensor
    value); it removes a negligible per-call recomputation and is required by
    torch's FLOP counter, which fails on parametrised modules."""
    from torch.nn.utils import parametrize
    for mod in model.modules():
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
    return model


def build_xlsr(pretrained: bool = False) -> Wav2Vec2Model:
    if pretrained:
        model = Wav2Vec2Model.from_pretrained(XLSR_ID)
        model.config.layerdrop = 0.0
        model.config.mask_time_prob = 0.0
    else:
        torch.manual_seed(0)
        cfg = dict(XLSR53_CONFIG)
        if os.environ.get("GREENAFI_TINY"):  # plumbing tests only: 24 thin layers
            cfg.update(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                       num_conv_pos_embedding_groups=4, conv_dim=(32,) * 7)
        model = Wav2Vec2Model(Wav2Vec2Config(**cfg))
    return bake_weight_norm(model).eval()


def truncate_xlsr(model: Wav2Vec2Model, n_layers: int) -> Wav2Vec2Model:
    """Return a copy that stops after transformer layer ``n_layers``.

    The final encoder LayerNorm is replaced by Identity, so ``last_hidden_state``
    of the truncated model equals ``hidden_states[n_layers]`` of the full model
    (the raw output of layer ``n_layers``), which is the representation this study reads.
    ``check_equivalence.py`` verifies this numerically.
    """
    cfg = copy.deepcopy(model.config)
    cfg.num_hidden_layers = n_layers
    m = bake_weight_norm(Wav2Vec2Model(cfg))
    missing, unexpected = m.load_state_dict(model.state_dict(), strict=False)
    assert not missing, missing  # only the dropped layers may be left over
    # only the dropped layers (and the pre-training mask vector, unused at inference) may be left over
    assert all(k.startswith("encoder.layers.") or k == "masked_spec_embed" for k in unexpected), unexpected
    m.encoder.layer_norm = nn.Identity()
    return m.eval()


def build_whisper(pretrained: bool = False) -> WhisperForConditionalGeneration:
    if pretrained:
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID)
    else:
        torch.manual_seed(0)
        model = WhisperForConditionalGeneration(WhisperConfig(**WHISPER_MEDIUM_CONFIG))
    return model.eval()


def whisper_features(audio: torch.Tensor) -> torch.Tensor:
    """Log-mel features exactly as Whisper requires them: padded to 30 s."""
    fe = WhisperFeatureExtractor()  # defaults = whisper-medium feature extractor
    feats = fe(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_features
    return feats


class FinetuneHead(nn.Module):
    """XLSR-53 + mean pooling + linear layer to the 12 phrases.

    Used only to cost the per-speaker fine-tuning baselines (the approach of
    Shor et al., 2019 / Green et al., 2021, adapted to a 12-way classifier).
    """

    def __init__(self, encoder: Wav2Vec2Model, n_classes: int = 12):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, n_classes)

    def forward(self, x):
        h = self.encoder(x).last_hidden_state
        return self.head(h.mean(dim=1))


class LoRALinear(nn.Module):
    """y = W x + (B A x) * alpha / r, with W frozen; A, B are the only trained weights."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def apply_lora(encoder: Wav2Vec2Model, r: int = 8, alpha: int = 16) -> Wav2Vec2Model:
    """LoRA (Hu et al., 2022) on the query and value projections of every attention layer."""
    for layer in encoder.encoder.layers:
        att = layer.attention
        att.q_proj = LoRALinear(att.q_proj, r, alpha)
        att.v_proj = LoRALinear(att.v_proj, r, alpha)
    return encoder


def set_trainable(model: FinetuneHead, mode: str, top_k: int = 4) -> int:
    """mode: 'full' (all transformer layers + head; CNN feature encoder frozen,
    as is standard for wav2vec 2.0 fine-tuning) or 'top' (only the top_k
    transformer layers + head). Returns the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    enc = model.encoder
    if mode == "full":
        for mod in (enc.feature_projection, enc.encoder):
            for p in mod.parameters():
                p.requires_grad = True
    elif mode == "top":
        for layer in list(enc.encoder.layers)[-top_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in enc.encoder.layer_norm.parameters():
            p.requires_grad = True
    elif mode == "lora":
        for n, p in enc.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
    else:
        raise ValueError(mode)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
