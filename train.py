"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
if os.environ.get("OPENCLAW_DISABLE_TORCH_COMPILE") == "1":
    def _torch_compile(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn
else:
    _torch_compile = torch.compile
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn_res.experimental.autograd import BlockAttentionResiduals
    HAS_FLASH_ATTN_RES = True
except Exception as exc:
    BlockAttentionResiduals = None
    HAS_FLASH_ATTN_RES = False
    FLASH_ATTN_RES_IMPORT_ERROR = exc

# OpenClaw fallback: kernels/FA3 is preferred, but this host already has a working
# CUDA torch install while uv is repairing. Fall back to PyTorch SDPA so training
# actually runs on the 3090s instead of idling.
try:
    if os.environ.get("OPENCLAW_FORCE_SDPA") == "1":
        raise RuntimeError("OPENCLAW_FORCE_SDPA=1")
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    HAS_FA3 = True
    print(f"Attention backend: {repo}", flush=True)
except Exception as exc:
    fa3 = None
    HAS_FA3 = False
    print(f"Attention backend: torch.scaled_dot_product_attention fallback ({type(exc).__name__}: {exc})", flush=True)

from prepare import MAX_SEQ_LEN, TIME_BUDGET as DEFAULT_TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

TIME_BUDGET = int(os.environ.get("TIME_BUDGET_SECONDS", DEFAULT_TIME_BUDGET))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name, default):
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name, default):
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


EXPERIMENT_LABEL = os.environ.get("EXPERIMENT_LABEL", "baseline")
USE_VALUE_EMBEDS = _env_bool("USE_VALUE_EMBEDS", True)
OPTIMIZER_KIND = os.environ.get("OPTIMIZER_KIND", "muon").strip().lower()
MLP_KIND = os.environ.get("MLP_KIND", "relu_squared").strip().lower()
FP32_ADAM_STATE = _env_bool("FP32_ADAM_STATE", False)
SANITY_REUSE_FIRST_BATCH = _env_bool("SANITY_REUSE_FIRST_BATCH", False)
FAILFAST_RANDOM_MARGIN = _env_float("FAILFAST_RANDOM_MARGIN", 0.04)
FAILFAST_MIN_PROGRESS = _env_float("FAILFAST_MIN_PROGRESS", 0.15)
FAILFAST_MIN_STEPS = _env_int("FAILFAST_MIN_STEPS", 80)
TRAIN_PROBE_BATCHES = _env_int("TRAIN_PROBE_BATCHES", 4)
GRAD_CLIP_NORM = _env_float("GRAD_CLIP_NORM", 0.0)
UNCOUNTED_WARMUP_STEPS = _env_int("UNCOUNTED_WARMUP_STEPS", 0)
FAILFAST_REGRESSION_MIN_STEPS = _env_int("FAILFAST_REGRESSION_MIN_STEPS", 40)
FAILFAST_REGRESSION_MIN_RISE = _env_float("FAILFAST_REGRESSION_MIN_RISE", 0.35)
FAILFAST_REGRESSION_PATIENCE_EVENTS = _env_int("FAILFAST_REGRESSION_PATIENCE_EVENTS", 3)
ATTN_RESIDUAL_MODE = os.environ.get("ATTN_RESIDUAL_MODE", "none").strip().lower()
ATTN_RESIDUAL_BACKEND = os.environ.get("ATTN_RESIDUAL_BACKEND", "package").strip().lower()
ATTN_RESIDUAL_BLOCK_SIZE = _env_int("ATTN_RESIDUAL_BLOCK_SIZE", 4)

# ---------------------------------------------------------------------------
# Lightweight run telemetry
# ---------------------------------------------------------------------------

RUN_ID = os.environ.get("AUTORESEARCH_RUN_ID") or time.strftime("transformer-paper-%Y%m%d-%H%M%S")
CAMPAIGN_ID = os.environ.get("AUTORESEARCH_CAMPAIGN_ID", "standalone")
CAMPAIGN_ROUND = _env_int("AUTORESEARCH_CAMPAIGN_ROUND", 0)
GPU_INDEX = _env_int("AUTORESEARCH_GPU_INDEX", 0)
RUN_SEED = _env_int("AUTORESEARCH_SEED", 42)
CODE_SHA = os.environ.get("AUTORESEARCH_CODE_SHA", "unknown")
METRICS_DIR = Path(os.environ.get("AUTORESEARCH_METRICS_DIR", "metrics"))
DASHBOARD_INGEST_URL = os.environ.get("DASHBOARD_INGEST_URL", "")
DASHBOARD_INGEST_TOKEN = os.environ.get("DASHBOARD_INGEST_TOKEN", "")
DASHBOARD_SITES_AUTH_TOKEN = os.environ.get("DASHBOARD_SITES_AUTH_TOKEN", "")
HOSTNAME = socket.gethostname()
CHECKPOINT_IF_BEST = _env_bool("OPENCLAW_CHECKPOINT_IF_BEST", False)
CHECKPOINT_BEST_VAL_BPB = _env_float("OPENCLAW_BEST_VAL_BPB", float("inf"))
CHECKPOINT_DIR = Path(os.environ.get("OPENCLAW_CHECKPOINT_DIR", "checkpoints"))
SAMPLE_PROMPT = os.environ.get("OPENCLAW_SAMPLE_PROMPT", "Once upon a time")
SAMPLE_MAX_NEW_TOKENS = _env_int("OPENCLAW_SAMPLE_TOKENS", 160)
SAMPLE_TEMPERATURE = _env_float("OPENCLAW_SAMPLE_TEMPERATURE", 0.8)
SAMPLE_TOP_K = _env_int("OPENCLAW_SAMPLE_TOP_K", 50)


def _post_event(event):
    if not DASHBOARD_INGEST_URL:
        return
    body = json.dumps(event).encode("utf-8")
    headers = {"content-type": "application/json"}
    if DASHBOARD_INGEST_TOKEN:
        headers["authorization"] = f"Bearer {DASHBOARD_INGEST_TOKEN}"
    if DASHBOARD_SITES_AUTH_TOKEN:
        headers["oai-sites-authorization"] = f"Bearer {DASHBOARD_SITES_AUTH_TOKEN}"
    request = urllib.request.Request(DASHBOARD_INGEST_URL, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            response.read()
    except Exception as exc:
        print(f"\ntelemetry_post_failed: {type(exc).__name__}: {exc}", flush=True)


def emit_event(kind, **payload):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "kind": kind,
        "run_id": RUN_ID,
        "campaign_id": CAMPAIGN_ID,
        "round": CAMPAIGN_ROUND,
        "gpu": GPU_INDEX,
        "seed": RUN_SEED,
        "code_sha": CODE_SHA,
        "experiment_label": EXPERIMENT_LABEL,
        "host": HOSTNAME,
        "time": time.time(),
        **payload,
    }
    with (METRICS_DIR / f"{RUN_ID}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    latest_path = METRICS_DIR / f"{RUN_ID}.latest.json"
    latest_tmp = METRICS_DIR / f".{RUN_ID}.{os.getpid()}.latest.tmp"
    latest_tmp.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
    latest_tmp.replace(latest_path)
    if DASHBOARD_INGEST_URL:
        if kind in ("start", "final", "failfast_random_loss", "failfast_probe_regression"):
            _post_event(event)
        else:
            threading.Thread(target=_post_event, args=(event,), daemon=True).start()

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),))
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return USE_VALUE_EMBEDS and layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if HAS_FA3:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # PyTorch SDPA can select flash attention on Ampere, but it cannot
            # honor the FA3-only sliding-window hint. Refuse that fake comparison.
            if window_size[0] != T:
                raise RuntimeError(
                    "Sliding-window attention requires FA3; use "
                    "WINDOW_PATTERN=LLLL with the PyTorch SDPA fallback."
                )
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
            y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if MLP_KIND == "swiglu":
            # A parameter-matched SwiGLU FFN uses roughly 8d/3 hidden units.
            # Round to an Ampere-friendly multiple without materially changing
            # the parameter budget relative to the 4d ReLU-squared baseline.
            self.hidden_dim = 128 * round((8 * config.n_embd / 3) / 128)
            self.c_fc = nn.Linear(config.n_embd, 2 * self.hidden_dim, bias=False)
            self.c_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        elif MLP_KIND == "relu_squared":
            self.hidden_dim = 4 * config.n_embd
            self.c_fc = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
            self.c_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        else:
            raise ValueError(f"Unknown MLP_KIND={MLP_KIND!r}; expected 'relu_squared' or 'swiglu'")

    def forward(self, x):
        x = self.c_fc(x)
        if MLP_KIND == "swiglu":
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value
        else:
            x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        if ATTN_RESIDUAL_MODE in ("block", "torch"):
            self.attn_res_pseudo_queries = nn.Parameter(torch.empty(config.n_layer + 1, config.n_embd))
        else:
            self.attn_res_pseudo_queries = None
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        if self.attn_res_pseudo_queries is not None:
            torch.nn.init.uniform_(self.attn_res_pseudo_queries, -s, s)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        attn_res_pseudo_queries = 0 if self.attn_res_pseudo_queries is None else self.attn_res_pseudo_queries.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + attn_res_pseudo_queries
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'attn_res_pseudo_queries': attn_res_pseudo_queries, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        optimizer_kind="muon"):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        attn_res_params = [] if self.attn_res_pseudo_queries is None else [self.attn_res_pseudo_queries]
        matrix_params = matrix_params + attn_res_params
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if optimizer_kind == "adamw_only":
            # Conservative diagnostic: remove Muon entirely while preserving the
            # same model/data/eval path. This tells us whether the U-shaped fixed
            # probe is optimizer-induced before trying architecture novelty.
            param_groups.append(dict(
                kind='adamw', params=matrix_params, lr=matrix_lr * dmodel_lr_scale,
                betas=adam_betas, eps=1e-10, weight_decay=weight_decay,
            ))
        elif optimizer_kind == "muon":
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        else:
            raise ValueError(f"Unknown OPTIMIZER_KIND={optimizer_kind!r}; expected 'muon' or 'adamw_only'")
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @staticmethod
    def _naive_attention_residual(values, pseudo_query):
        keys = norm(values.float())
        logits = torch.einsum("d,nbtd->nbt", pseudo_query.float(), keys)
        weights = logits.softmax(dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, values.float()).to(values.dtype)

    def _forward_attention_residuals_torch(self, x, cos_sin):
        blocks = [x]
        input_dtype = x.dtype
        for i, block in enumerate(self.transformer.h):
            layer_input = self._naive_attention_residual(torch.stack(blocks, dim=0), self.attn_res_pseudo_queries[i])
            update = block(layer_input.to(input_dtype), None, cos_sin, self.window_sizes[i])
            if i % ATTN_RESIDUAL_BLOCK_SIZE == 0:
                blocks.append(update)
            else:
                blocks[-1] = blocks[-1] + update
        return self._naive_attention_residual(torch.stack(blocks, dim=0), self.attn_res_pseudo_queries[-1]).to(input_dtype)

    def _forward_attention_residuals_package(self, x, cos_sin):
        if not HAS_FLASH_ATTN_RES:
            raise RuntimeError(f"flash-attn-res import failed: {FLASH_ATTN_RES_IMPORT_ERROR!r}")

        class LayerAdapter:
            def __init__(self, module, window_size):
                self.module = module
                self.window_size = window_size

            def __call__(self, layer_input):
                # flash-attn-res recomputes layers in its custom backward outside
                # the outer training autocast context; restore autocast so Linear
                # sees bf16 activations with fp32 weights just like the baseline.
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.module(layer_input, None, cos_sin, self.window_size)

            def parameters(self):
                return self.module.parameters()

        layers = [LayerAdapter(block, self.window_sizes[i]) for i, block in enumerate(self.transformer.h)]
        flat_layer_params = tuple(p for block in self.transformer.h for p in block.parameters())
        return BlockAttentionResiduals.apply(
            x,
            self.attn_res_pseudo_queries,
            layers,
            ATTN_RESIDUAL_BLOCK_SIZE,
            torch.finfo(torch.float32).eps,
            *flat_layer_params,
        )

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        if ATTN_RESIDUAL_MODE in ("block", "torch"):
            if ATTN_RESIDUAL_BACKEND == "torch" or ATTN_RESIDUAL_MODE == "torch":
                x = self._forward_attention_residuals_torch(x, cos_sin)
            else:
                x = self._forward_attention_residuals_package(x, cos_sin)
        else:
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@_torch_compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    if os.environ.get("OPENCLAW_DISABLE_TORCH_COMPILE") == "1":
        step_v = float(step_t.item()) if hasattr(step_t, "item") else float(step_t)
        lr_v = float(lr_t.item()) if hasattr(lr_t, "item") else float(lr_t)
        beta1_v = float(beta1_t.item()) if hasattr(beta1_t, "item") else float(beta1_t)
        beta2_v = float(beta2_t.item()) if hasattr(beta2_t, "item") else float(beta2_t)
        eps_v = float(eps_t.item()) if hasattr(eps_t, "item") else float(eps_t)
        wd_v = float(wd_t.item()) if hasattr(wd_t, "item") else float(wd_t)
        p.mul_(1 - lr_v * wd_v)
        exp_avg.lerp_(grad, 1 - beta1_v)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_v)
        bias1 = 1 - beta1_v ** step_v
        bias2 = 1 - beta2_v ** step_v
        denom = (exp_avg_sq / bias2).sqrt() + eps_v
        step_size = lr_v / bias1
        p.add_(exp_avg / denom, alpha=-step_size)
        return
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@_torch_compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    _eager = os.environ.get("OPENCLAW_DISABLE_TORCH_COMPILE") == "1"
    if _eager:
        momentum = float(momentum_t.item()) if hasattr(momentum_t, "item") else float(momentum_t)
        lr_v = float(lr_t.item()) if hasattr(lr_t, "item") else float(lr_t)
        wd_v = float(wd_t.item()) if hasattr(wd_t, "item") else float(wd_t)
        beta2_v = float(beta2_t.item()) if hasattr(beta2_t, "item") else float(beta2_t)
    else:
        momentum = momentum_t.to(stacked_grads.dtype)
    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.float() if os.environ.get("OPENCLAW_DISABLE_TORCH_COMPILE") == "1" else g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_v if _eager else beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_v if _eager else lr_t.to(g.dtype)
    wd = wd_v if _eager else wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                if FP32_ADAM_STATE and p.dtype != torch.float32:
                    state['master_param'] = p.detach().float().clone()
                state_param = state.get('master_param', p)
                state['exp_avg'] = torch.zeros_like(state_param)
                state['exp_avg_sq'] = torch.zeros_like(state_param)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            update_param = state.get('master_param', p)
            update_grad = grad.float() if update_param.dtype == torch.float32 else grad
            adamw_step_fused(update_param, update_grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            if update_param is not p:
                p.copy_(update_param)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        if hasattr(torch, "_foreach_copy_"):
            torch._foreach_copy_(params, list(stacked_params.unbind(0)))
        else:
            for p_dst, p_src in zip(params, stacked_params.unbind(0)):
                p_dst.copy_(p_src)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture. Defaults match the repo baseline; env overrides let
# OpenClaw run clean experiment arms without one-off source edits.
ASPECT_RATIO = _env_int("ASPECT_RATIO", 64)       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = _env_int("HEAD_DIM", 128)              # target head dimension for attention
WINDOW_PATTERN = os.environ.get("WINDOW_PATTERN", "SSSL") # L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = _env_int("TOTAL_BATCH_SIZE", 2**19) # ~524K tokens per optimizer step
EMBEDDING_LR = _env_float("EMBEDDING_LR", 0.6)      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = _env_float("UNEMBEDDING_LR", 0.004) # learning rate for lm_head (Adam)
MATRIX_LR = _env_float("MATRIX_LR", 0.04)           # learning rate for matrix parameters (Muon)
SCALAR_LR = _env_float("SCALAR_LR", 0.5)            # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.2)      # cautious weight decay for Muon
ADAM_BETAS = (
    _env_float("ADAM_BETA1", 0.8),
    _env_float("ADAM_BETA2", 0.95),
)
WARMUP_RATIO = _env_float("WARMUP_RATIO", 0.0)      # fraction of time budget for LR warmup
WARMDOWN_RATIO = _env_float("WARMDOWN_RATIO", 0.5)  # fraction of time budget for LR warmdown
FINAL_LR_FRAC = _env_float("FINAL_LR_FRAC", 0.0)    # final LR as fraction of initial

# Model size
DEPTH = _env_int("DEPTH", 8)                         # number of transformer layers
DEVICE_BATCH_SIZE = _env_int("DEVICE_BATCH_SIZE", 128)  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(RUN_SEED)
torch.cuda.manual_seed(RUN_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
_amp_dtype_name = os.environ.get("AMP_DTYPE", "auto").lower()
if _amp_dtype_name in ("fp16", "float16", "half"):
    _amp_dtype = torch.float16
elif _amp_dtype_name in ("bf16", "bfloat16"):
    _amp_dtype = torch.bfloat16
else:
    _amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=_amp_dtype)
print(f"Experiment label: {EXPERIMENT_LABEL}", flush=True)
print(f"Use value embeddings: {USE_VALUE_EMBEDS}", flush=True)
print(f"Optimizer kind: {OPTIMIZER_KIND}", flush=True)
print(f"FP32 Adam state/master weights: {FP32_ADAM_STATE}", flush=True)
print(f"MLP kind: {MLP_KIND}", flush=True)
print(f"Attention residual mode/backend/block: {ATTN_RESIDUAL_MODE}/{ATTN_RESIDUAL_BACKEND}/{ATTN_RESIDUAL_BLOCK_SIZE}", flush=True)
print(f"flash-attn-res available: {HAS_FLASH_ATTN_RES}", flush=True)
print(f"AMP dtype: {_amp_dtype}", flush=True)
REFERENCE_PEAK_FLOPS = _env_float("REFERENCE_PEAK_FLOPS", 71.0e12)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")
emit_event(
    "start",
    model_family="transformer-paper",
    experiment_label=EXPERIMENT_LABEL,
    goal="Reimplement and study the core Attention Is All You Need Transformer ideas inside autoresearch.",
    config=asdict(config),
    hparams=dict(
        use_value_embeds=USE_VALUE_EMBEDS,
        optimizer_kind=OPTIMIZER_KIND,
        fp32_adam_state=FP32_ADAM_STATE,
        mlp_kind=MLP_KIND,
        aspect_ratio=ASPECT_RATIO,
        head_dim=HEAD_DIM,
        embedding_lr=EMBEDDING_LR,
        unembedding_lr=UNEMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        scalar_lr=SCALAR_LR,
        weight_decay=WEIGHT_DECAY,
        adam_betas=ADAM_BETAS,
        warmup_ratio=WARMUP_RATIO,
        warmdown_ratio=WARMDOWN_RATIO,
        final_lr_frac=FINAL_LR_FRAC,
        amp_dtype=str(_amp_dtype),
        sanity_reuse_first_batch=SANITY_REUSE_FIRST_BATCH,
        failfast_random_margin=FAILFAST_RANDOM_MARGIN,
        failfast_min_progress=FAILFAST_MIN_PROGRESS,
        failfast_min_steps=FAILFAST_MIN_STEPS,
        train_probe_batches=TRAIN_PROBE_BATCHES,
        grad_clip_norm=GRAD_CLIP_NORM,
        uncounted_warmup_steps=UNCOUNTED_WARMUP_STEPS,
        attn_residual_mode=ATTN_RESIDUAL_MODE,
        attn_residual_backend=ATTN_RESIDUAL_BACKEND,
        attn_residual_block_size=ATTN_RESIDUAL_BLOCK_SIZE,
    ),
    time_budget_seconds=TIME_BUDGET,
    max_seq_len=MAX_SEQ_LEN,
    total_batch_size=TOTAL_BATCH_SIZE,
    device_batch_size=DEVICE_BATCH_SIZE,
    seed=RUN_SEED,
    attention_backend="fa3" if HAS_FA3 else "pytorch_sdpa",
    torch_version=torch.__version__,
    cuda_version=torch.version.cuda,
    device_name=torch.cuda.get_device_name(),
)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    optimizer_kind=OPTIMIZER_KIND,
)

model = _torch_compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

# Keep a small fixed train probe so training-loss telemetry is comparable over
# time. The previous plotted "train_loss" was an online, pre-update EMA over a
# moving dataloader stream; that can look U-shaped when later batches are harder
# even if the model is improving. Probe batches are consumed once, cloned, and
# then excluded from the optimization stream.
train_probe_batches = []
for _ in range(max(1, TRAIN_PROBE_BATCHES)):
    px, py, epoch = next(train_loader)
    train_probe_batches.append((px.detach().clone(), py.detach().clone()))

if SANITY_REUSE_FIRST_BATCH:
    x, y = train_probe_batches[0]
else:
    x, y, epoch = next(train_loader)
if SANITY_REUSE_FIRST_BATCH:
    print("Sanity mode: reusing the first probe batch every microstep (tiny overfit test)", flush=True)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Random-loss baseline ln(vocab): {math.log(vocab_size):.6f}; fail-fast margin: {FAILFAST_RANDOM_MARGIN:.3f}")
print(f"Fixed train probe batches: {len(train_probe_batches)}")
print(f"Gradient clipping: {GRAD_CLIP_NORM if GRAD_CLIP_NORM > 0 else 'disabled'}")
print(f"Uncounted compiler warm-up steps: {UNCOUNTED_WARMUP_STEPS}")
print(
    "Fixed-probe regression fail-fast: "
    f"rise>{FAILFAST_REGRESSION_MIN_RISE:.3f} after step {FAILFAST_REGRESSION_MIN_STEPS} "
    f"for {FAILFAST_REGRESSION_PATIENCE_EVENTS} emitted events"
)

@torch.no_grad()
def evaluate_train_probe_loss():
    was_training = model.training
    model.eval()
    losses = []
    for px, py in train_probe_batches:
        with autocast_ctx:
            probe_loss = model(px, py)
        losses.append(float(probe_loss.item()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def model_diagnostics():
    """Small, stable signals that make optimizer collapse diagnosable."""
    wte = model.transformer.wte.weight.float()
    lm_head = model.lm_head.weight.float()
    return {
        "wte_rms": float(wte.square().mean().sqrt().item()),
        "lm_head_rms": float(lm_head.square().mean().sqrt().item()),
        "resid_lambda_abs_max": float(model.resid_lambdas.abs().max().item()),
        "x0_lambda_abs_max": float(model.x0_lambdas.abs().max().item()),
        "x0_lambda_min": float(model.x0_lambdas.min().item()),
        "x0_lambda_max": float(model.x0_lambdas.max().item()),
    }

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
best_probe_train_loss = None
best_probe_train_step = None
probe_regression_events = 0
total_training_time = 0
step = 0
last_emit_time = 0.0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    micro_loss_sum = 0.0
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        micro_loss_sum += float(train_loss.item())
        loss = loss / grad_accum_steps
        loss.backward()
        if not SANITY_REUSE_FIRST_BATCH:
            x, y, epoch = next(train_loader)

    capture_grad_norm = GRAD_CLIP_NORM > 0 or step == 0 or time.time() - last_emit_time >= 15
    grad_norm = None
    if capture_grad_norm:
        max_norm = GRAD_CLIP_NORM if GRAD_CLIP_NORM > 0 else float("inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = micro_loss_sum / max(1, grad_accum_steps)

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL: loss exploded or became NaN")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step >= UNCOUNTED_WARMUP_STEPS:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / REFERENCE_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    now = time.time()
    random_loss = math.log(vocab_size)
    should_emit = step == 0 or now - last_emit_time >= 15 or remaining <= 0
    probe_train_loss = evaluate_train_probe_loss() if should_emit else None
    failfast_loss = probe_train_loss
    if failfast_loss is not None:
        if best_probe_train_loss is None or failfast_loss < best_probe_train_loss:
            best_probe_train_loss = failfast_loss
            best_probe_train_step = step
            probe_regression_events = 0
        elif (
            not SANITY_REUSE_FIRST_BATCH
            and step >= FAILFAST_REGRESSION_MIN_STEPS
            and best_probe_train_step is not None
            and best_probe_train_step < step
            and failfast_loss > best_probe_train_loss + FAILFAST_REGRESSION_MIN_RISE
        ):
            probe_regression_events += 1
        else:
            probe_regression_events = 0
    too_close_to_random = (
        failfast_loss is not None
        and not SANITY_REUSE_FIRST_BATCH
        and step >= FAILFAST_MIN_STEPS
        and progress >= FAILFAST_MIN_PROGRESS
        and failfast_loss > random_loss - FAILFAST_RANDOM_MARGIN
    )
    if too_close_to_random:
        emit_event(
            "failfast_random_loss",
            step=step,
            progress=pct_done,
            train_loss=failfast_loss,
            probe_train_loss=probe_train_loss,
            online_train_loss=debiased_smooth_loss,
            random_loss=random_loss,
            margin=FAILFAST_RANDOM_MARGIN,
            lr_multiplier=lrm,
            total_training_seconds=total_training_time,
        )
        print(f"\nFAIL: fixed-probe train loss {failfast_loss:.6f} remains within {FAILFAST_RANDOM_MARGIN:.3f} nats of random baseline {random_loss:.6f}", flush=True)
        exit(2)

    regressed_from_best = (
        failfast_loss is not None
        and best_probe_train_loss is not None
        and probe_regression_events >= FAILFAST_REGRESSION_PATIENCE_EVENTS
    )
    if regressed_from_best:
        emit_event(
            "failfast_probe_regression",
            step=step,
            progress=pct_done,
            train_loss=failfast_loss,
            probe_train_loss=probe_train_loss,
            online_train_loss=debiased_smooth_loss,
            best_probe_train_loss=best_probe_train_loss,
            best_probe_train_step=best_probe_train_step,
            rise_from_best=failfast_loss - best_probe_train_loss,
            min_rise=FAILFAST_REGRESSION_MIN_RISE,
            patience_events=FAILFAST_REGRESSION_PATIENCE_EVENTS,
            lr_multiplier=lrm,
            total_training_seconds=total_training_time,
        )
        print(
            "\nFAIL: fixed-probe train loss regressed "
            f"from best {best_probe_train_loss:.6f} at step {best_probe_train_step} "
            f"to {failfast_loss:.6f} at step {step}",
            flush=True,
        )
        exit(2)

    if should_emit:
        emit_event(
            "step",
            step=step,
            progress=pct_done,
            train_loss=probe_train_loss,
            probe_train_loss=probe_train_loss,
            online_train_loss=debiased_smooth_loss,
            raw_micro_loss=train_loss_f,
            random_loss=random_loss,
            best_probe_train_loss=best_probe_train_loss,
            best_probe_train_step=best_probe_train_step,
            probe_regression_events=probe_regression_events,
            lr_multiplier=lrm,
            step_seconds=dt,
            tokens_per_second=tok_per_sec,
            mfu_percent=mfu,
            grad_norm=grad_norm,
            model_diagnostics=model_diagnostics(),
            epoch=epoch,
            remaining_seconds=remaining,
            total_training_seconds=total_training_time,
        )
        last_emit_time = now

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up. Compiler-only warm-up steps can be excluded explicitly, while
    # eager runs count useful work from the first step.
    if step > UNCOUNTED_WARMUP_STEPS and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

counted_steps = max(0, step - UNCOUNTED_WARMUP_STEPS)
total_tokens = counted_steps * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * counted_steps / total_training_time / REFERENCE_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
emit_event(
    "final",
    val_bpb=val_bpb,
    training_seconds=total_training_time,
    total_seconds=t_end - t_start,
    startup_seconds=startup_time,
    peak_vram_mb=peak_vram_mb,
    mfu_percent=steady_state_mfu,
    total_tokens_m=total_tokens / 1e6,
    num_steps=step,
    counted_steps=counted_steps,
    num_params_m=num_params / 1e6,
    depth=DEPTH,
    experiment_label=EXPERIMENT_LABEL,
    use_value_embeds=USE_VALUE_EMBEDS,
    window_pattern=WINDOW_PATTERN,
    optimizer_kind=OPTIMIZER_KIND,
    fp32_adam_state=FP32_ADAM_STATE,
    mlp_kind=MLP_KIND,
    attn_residual_mode=ATTN_RESIDUAL_MODE,
    attn_residual_backend=ATTN_RESIDUAL_BACKEND,
    attn_residual_block_size=ATTN_RESIDUAL_BLOCK_SIZE,
)


def _checkpoint_model():
    return getattr(model, "_orig_mod", model)


@torch.no_grad()
def _sample_text(max_new_tokens=SAMPLE_MAX_NEW_TOKENS):
    if max_new_tokens <= 0:
        return ""
    base_model = _checkpoint_model()
    was_training = base_model.training
    base_model.eval()
    prompt_ids = tokenizer.encode(SAMPLE_PROMPT, prepend=tokenizer.get_bos_token_id())
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    temperature = max(1e-6, SAMPLE_TEMPERATURE)
    for _ in range(max_new_tokens):
        ctx = ids[:, -MAX_SEQ_LEN:]
        with autocast_ctx:
            logits = base_model(ctx)[:, -1, :].float() / temperature
        if SAMPLE_TOP_K > 0 and SAMPLE_TOP_K < logits.size(-1):
            values, _ = torch.topk(logits, SAMPLE_TOP_K)
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        ids = torch.cat([ids, next_id], dim=1)
    if was_training:
        base_model.train()
    out_ids = ids[0].tolist()
    if out_ids and out_ids[0] == tokenizer.get_bos_token_id():
        out_ids = out_ids[1:]
    return tokenizer.decode(out_ids)


if CHECKPOINT_IF_BEST and val_bpb < CHECKPOINT_BEST_VAL_BPB:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{RUN_ID}-val{val_bpb:.6f}"
    ckpt_path = CHECKPOINT_DIR / f"{stem}.pt"
    sample_path = CHECKPOINT_DIR / f"{stem}.sample.txt"
    base_model = _checkpoint_model()
    torch.save(
        {
            "run_id": RUN_ID,
            "val_bpb": val_bpb,
            "best_val_bpb_before_run": CHECKPOINT_BEST_VAL_BPB,
            "config": asdict(config),
            "hparams": dict(
                use_value_embeds=USE_VALUE_EMBEDS,
                optimizer_kind=OPTIMIZER_KIND,
                fp32_adam_state=FP32_ADAM_STATE,
                mlp_kind=MLP_KIND,
                uncounted_warmup_steps=UNCOUNTED_WARMUP_STEPS,
                aspect_ratio=ASPECT_RATIO,
                head_dim=HEAD_DIM,
                embedding_lr=EMBEDDING_LR,
                unembedding_lr=UNEMBEDDING_LR,
                matrix_lr=MATRIX_LR,
                scalar_lr=SCALAR_LR,
                weight_decay=WEIGHT_DECAY,
                adam_betas=ADAM_BETAS,
                warmup_ratio=WARMUP_RATIO,
                warmdown_ratio=WARMDOWN_RATIO,
                final_lr_frac=FINAL_LR_FRAC,
                total_batch_size=TOTAL_BATCH_SIZE,
                device_batch_size=DEVICE_BATCH_SIZE,
                time_budget_seconds=TIME_BUDGET,
                attn_residual_mode=ATTN_RESIDUAL_MODE,
                attn_residual_backend=ATTN_RESIDUAL_BACKEND,
                attn_residual_block_size=ATTN_RESIDUAL_BLOCK_SIZE,
            ),
            "state_dict": base_model.state_dict(),
        },
        ckpt_path,
    )
    sample = _sample_text()
    sample_path.write_text(sample, encoding="utf-8", errors="replace")
    print(f"checkpoint_saved: {ckpt_path}")
    print(f"sample_saved:     {sample_path}")
    emit_event("checkpoint", checkpoint=str(ckpt_path), sample=str(sample_path), sample_prompt=SAMPLE_PROMPT)
