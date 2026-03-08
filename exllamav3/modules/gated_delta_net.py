from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import get_for_device, to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
from .gated_rmsnorm import GatedRMSNorm
from ..cache import CacheableState
from ..util.tensor import g_tensor_cache
from ..util import profile_opt

"""
causal_conv1d wrappers and fallback functions
"""

def causal_conv1d_update_function_torch(
    x,
    conv_state,
    weight,
    bias = None,
):
    bsz, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]

    y = torch.cat([conv_state, x], dim = -1).to(weight.dtype)
    conv_state.copy_(y[:, :, -state_len:])
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


def causal_conv1d_fwd_function_torch(
    x,
    weight,
    bias,
):
    # Differs from Qwen3-Next Transformers impl. but corresponds better to causal_conv1d which uses zeros
    # as the initial state
    bsz, dim, seq_len = x.shape
    zero_state = torch.zeros((bsz, dim, weight.shape[-1]), dtype = x.dtype, device = x.device)

    y = torch.cat([zero_state, x], dim = -1).to(weight.dtype)
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


def causal_conv1d_update_function_cu(
    x,
    conv_state,
    weight,
    bias = None,
):
    y = torch.empty_like(x)
    causal_conv1d_cuda.causal_conv1d_update(x, conv_state, weight, bias, y, True, None, None)
    return y


def causal_conv1d_fwd_function_cu(
    x,
    weight,
    bias,
):
    y = torch.empty_like(x)
    causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, None, None, y, None, True)
    return y


try:
    import causal_conv1d_cuda
    causal_conv1d_update_function = causal_conv1d_update_function_cu
    causal_conv1d_fwd_function = causal_conv1d_fwd_function_cu
except ModuleNotFoundError:
    causal_conv1d_update_function = causal_conv1d_update_function_torch
    causal_conv1d_fwd_function = causal_conv1d_fwd_function_torch

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ModuleNotFoundError:
    chunk_gated_delta_rule = None

"""
fla wrapper, reduce overhead by bypassing input_guard and torch custom ops stuff
"""

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

    scale = k.shape[-1] ** -0.5
    with torch.cuda.device(q.device.index):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v.contiguous(),
            g,
            None,
            None,
            beta,
            scale,
            initial_state.contiguous() if initial_state is not None else None,
            output_final_state,
            use_qk_l2norm_in_kernel,
            None,
        )
    return o, final_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = 1 / torch.sqrt(
            (x * x).sum(dim = dim, keepdim = True)
            + eps
        )
        return x * inv_norm

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape

    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()

    for i in range(sequence_length):
        q_t = query[:, i, :]
        k_t = key[:, i, :]
        v_t = value[:, i, :]
        g_t = g[:, i, :].exp().unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)
        kv_mem = last_recurrent_state * k_t.unsqueeze(-1)
        kv_mem = kv_mem.sum(dim = -2)
        v_t = v_t - kv_mem * g_t
        upd = k_t.unsqueeze(-1) * v_t.unsqueeze(-2) * beta_t.unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t.unsqueeze(-1) + upd
        core_attn_out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2) * scale

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


class GDN_RecurrentState(CacheableState):
    def __init__(
        self,
        position: int | None = 0,
        positions: list[int] | None = None,
        last_conv_state: torch.Tensor = None,
        last_recurrent_state: torch.Tensor = None,
        batched = False
    ):
        super().__init__()
        self.position = position
        self.positions = positions
        self.last_conv_state = last_conv_state
        self.last_recurrent_state = last_recurrent_state
        self.batched = batched

    @override
    def stash(self):
        # TODO: Option to preallocate and pin space for stashed states
        return GDN_RecurrentState(
            self.position,
            self.positions,
            self.last_conv_state.cpu() if self.last_conv_state is not None else None,
            self.last_recurrent_state.cpu() if self.last_recurrent_state is not None else None,
        )

    @override
    def unstash(self, device):
        return GDN_RecurrentState(
            self.position,
            self.positions,
            self.last_conv_state.to(device, non_blocking = True) if self.last_conv_state is not None else None,
            self.last_recurrent_state.to(device, non_blocking = True) if self.last_recurrent_state is not None else None,
        )

    @override
    def get_size(self):
        if self.last_conv_state is None:
            return 0
        return (
            self.last_conv_state.element_size() * self.last_conv_state.numel() +
            self.last_recurrent_state.element_size() * self.last_recurrent_state.numel()
        )

    def collect_batch(self, batch: list[GDN_RecurrentState]):
        lcs = torch.cat([b.last_conv_state for b in batch], dim = 0)
        lrs = torch.cat([b.last_recurrent_state for b in batch], dim = 0)
        positions = [b.position for b in batch]
        return GDN_RecurrentState(None, positions, lcs, lrs, True)

    def distribute_batch(self, batch: list[GDN_RecurrentState]):
        for i, b in enumerate(batch):
            b.last_conv_state.copy_(self.last_conv_state[i:i+1, ...])
            b.last_recurrent_state.copy_(self.last_recurrent_state[i:i+1, ...])
            b.position = self.positions[i]


def prepare_for_recurrence(input_ids: torch.Tensor, params: dict, model) -> torch.Tensor:
    """
    Add linear attn parameters to state

    batch_shape: tuple of (bsz, _)
    past_len: int (default: 0)

    *OR*

    cache_seqlens: shape (bsz)
    """
    batch_shape = params.get("batch_shape")
    cache_seqlens = params.get("cache_seqlens")

    if batch_shape is not None:
        bsz, _ = batch_shape
        past_len = params.get("past_len", 0)
        if past_len > 0:
            rs = params.get("recurrent_states")
            if rs is None:
                raise ValueError(f"Past length given, but no previous state for linear attn in params")
            for k, v in rs.items():
                if not v.batched and v.position != past_len:
                    raise ValueError(f"recurrent states don't match input past_len")
        else:
            rl = model.get_recurrent_layers()
            rs = {attn.layer_idx: GDN_RecurrentState() for attn in rl}
            params["recurrent_states"] = rs

    elif cache_seqlens is not None:
        # (Empty) states must be provided with cache_seqlens
        pass

    else:
        if "recurrent_states" in params:
            raise ValueError(f"recurrent_states given without bsz and seqlens")


class GatedDeltaNet(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        key_a_log: str | None = None,
        key_dt_bias: str | None = None,
        key_conv1d: str | None = None,
        key_fused_ba: str | None = None,
        key_fused_qkvz: str | None = None,
        key_qkv: str | None = None,
        key_z: str | None = None,
        key_b: str | None = None,
        key_a: str | None = None,
        key_norm: str | None = None,
        key_o: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        qkvz_proj: Linear | None = None,
        qkv_proj: Linear | None = None,
        z_proj: Linear | None = None,
        ba_proj: Linear | None = None,
        b_proj: Linear | None = None,
        a_proj: Linear | None = None,
        o_proj: Linear | None = None,
        norm: GatedRMSNorm | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "GatedDeltaNet"

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_v_groups = num_v_heads // num_k_heads
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads

        self.out_dtype = out_dtype

        self.fdim_qkvz = 2 * self.num_k_heads * self.k_head_dim + 2 * self.num_v_heads * self.v_head_dim
        self.fdim_ba = 2 * self.num_v_heads
        self.fdim_qkv = 2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim

        if key_qkv or key_z:
            assert key_qkv and key_z, \
                "GatedDeltaNet split qkv/z projections require both key_qkv and key_z"
        if key_b or key_a:
            assert key_b and key_a, \
                "GatedDeltaNet split b/a projections require both key_b and key_a"

        if qkvz_proj is not None:
            self.qkvz_proj = qkvz_proj
            self.register_submodule(self.qkvz_proj)
        elif key_fused_qkvz:
            self.qkvz_proj = Linear(config, f"{key}.{key_fused_qkvz}", hidden_size, self.fdim_qkvz, qmap = qmap + ".input", out_dtype = torch.float)
            self.register_submodule(self.qkvz_proj)
        else:
            self.qkvz_proj = None

        if qkv_proj is not None:
            self.qkv_proj = qkv_proj
            self.z_proj = z_proj
            self.register_submodule(self.qkv_proj)
            if self.z_proj is not None:
                self.register_submodule(self.z_proj)
        elif key_qkv:
            self.qkv_proj = Linear(config, f"{key}.{key_qkv}", hidden_size, self.fdim_qkv, qmap = qmap + ".input", out_dtype = torch.float)
            self.z_proj = Linear(config, f"{key}.{key_z}", hidden_size, self.v_dim, qmap = qmap + ".input", out_dtype = torch.float)
            self.register_submodule(self.qkv_proj)
            self.register_submodule(self.z_proj)
        else:
            self.qkv_proj = None
            self.z_proj = None

        if ba_proj is not None:
            self.ba_proj = ba_proj
            self.register_submodule(self.ba_proj)
        elif key_fused_ba:
            self.ba_proj = Linear(config, f"{key}.{key_fused_ba}", hidden_size, self.fdim_ba, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.register_submodule(self.ba_proj)
        else:
            self.ba_proj = None

        if b_proj is not None:
            self.b_proj = b_proj
            self.a_proj = a_proj
            self.register_submodule(self.b_proj)
            if self.a_proj is not None:
                self.register_submodule(self.a_proj)
        elif key_b:
            self.b_proj = Linear(config, f"{key}.{key_b}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.a_proj = Linear(config, f"{key}.{key_a}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        else:
            self.b_proj = None
            self.a_proj = None

        if o_proj is not None:
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)
        else:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                self.v_head_dim * self.num_v_heads,
                hidden_size,
                qmap = qmap + ".output",
                out_dtype = self.out_dtype
            )
            self.register_submodule(self.o_proj)

        if norm is not None:
            self.norm = norm
            self.register_submodule(self.norm)
        else:
            self.norm = GatedRMSNorm(config, f"{key}.{key_norm}", self.rms_norm_eps, out_dtype = torch.half)
            self.register_submodule(self.norm)

        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.key_a_log = f"{key}.{key_a_log}" if key_a_log else None
        self.key_dt_bias = f"{key}.{key_dt_bias}" if key_dt_bias else None
        self.key_conv1d_weight = f"{key}.{key_conv1d}.weight" if key_conv1d else None
        self.key_conv1d_bias = f"{key}.{key_conv1d}.bias" if key_conv1d else None

        self.conv_dim = self.k_head_dim * self.num_k_heads

        self.caps.update({
            "recurrent_cache": True
        })

        self.bc = None
        self.bsz1_pa_args = []
        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        if self.qkvz_proj is not None:
            return [[self.qkvz_proj.optimizer_targets()]]

        targets = []
        if self.qkv_proj is not None:
            targets += self.qkv_proj.optimizer_targets()
        if self.z_proj is not None:
            targets += self.z_proj.optimizer_targets()
        return [targets]


    def load_local(self, device, **kwargs):
        # Skip load_local for zero heads in TP mode
        if self.num_k_heads == 0:
            return

        is_quantized = (
            self.qkvz_proj is not None and self.qkvz_proj.quant_format_id() == "exl3" and
            self.ba_proj is not None and self.ba_proj.quant_format_id() is None and
            self.o_proj is not None and self.o_proj.quant_format_id() == "exl3"
        )

        # Skip bound class in TP mode - it's incompatible with split tensors
        if is_quantized and not self.tp_reduce:
            self.bsz1_pa_args = [
                (device, (1, self.fdim_qkv, 1), torch.bfloat16),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "a"),
                (device, (1, 1, self.num_v_heads), torch.bfloat16),
                (device, (1, 1, self.num_v_heads), torch.float),
                (device, (1, 1, self.fdim_qkvz), torch.float),
                (device, (1, 1, self.fdim_ba), torch.float),
                (device, (1, self.fdim_qkv, self.conv_kernel_size + 1), torch.bfloat16, "a"),
                (device, (1, self.fdim_qkv, 2), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads * self.v_head_dim), torch.half),
            ]

            self.bc = ext.BC_GatedDeltaNet(
                *(g_tensor_cache.get(*arg) for arg in self.bsz1_pa_args),
                self.qkvz_proj.inner.bc,
                self.ba_proj.inner.bc,
                self.dt_bias,
                self.a_log,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                self.conv1d_weight,
                self.conv1d_bias,
                self.norm.bc,
                self.o_proj.inner.bc
            )

    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device)
        if self.key_a_log is not None:
            self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device, optional = False, allow_bf16 = True)
        if self.key_dt_bias is not None:
            self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device, optional = False, allow_bf16 = True)
        if self.key_conv1d_weight is not None:
            self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = False, allow_bf16 = True)
        if self.key_conv1d_bias is not None:
            self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True)
        if self.norm is not None:
            self.norm.load(device, **kwargs)
        self.load_local(device, **kwargs)

    @override
    def unload(self):
        if self.bc is not None:
            for arg in self.bsz1_pa_args:
                g_tensor_cache.drop(*arg)
            self.bc = None
            self.bsz1_pa_args = []
        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.norm.unload()
        super().unload()


    def split_fused_inputs(self, mixed_qkvz, mixed_ba):
        # mixed_qkvz and mixed_ba have same (bsz, seqlen)
        # both are contiguous
        bsz, seqlen, _ = mixed_qkvz.shape

        mixed_qkvz = mixed_qkvz.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.k_head_dim + 2 * self.v_head_dim * self.num_v_heads // self.num_k_heads,
        )
        mixed_ba = mixed_ba.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.num_v_heads // self.num_k_heads
        )

        split_arg_list_qkvz = [
            self.k_head_dim,
            self.k_head_dim,
            (self.num_v_groups * self.v_head_dim),
            (self.num_v_groups * self.v_head_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]
        q, k, v, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim = 3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim = 3)

        q = q.reshape(bsz, seqlen, -1)
        k = k.reshape(bsz, seqlen, -1)
        v = v.reshape(bsz, seqlen, -1)
        z = z.reshape(bsz, seqlen, -1, self.v_head_dim)
        b = b.reshape(bsz, seqlen, self.num_v_heads)
        a = a.reshape(bsz, seqlen, self.num_v_heads)
        mixed_qkv = torch.cat((q, k, v), dim = -1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        return mixed_qkv, z, b, a


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        # Handle zero heads in TP mode (some workers may get 0 heads when splitting)
        if self.num_k_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(x, False)
            return to2(x, out_dtype, self.out_dtype)
        bsz, seqlen, _ = x.shape

        # Previous state
        rs = params.get("recurrent_states")
        if rs is not None:
            rs = rs[self.layer_idx]
            conv_state = rs.last_conv_state if rs.last_conv_state is not None else \
                torch.zeros((bsz, self.fdim_qkv, self.conv_kernel_size), dtype = torch.bfloat16, device = x.device)
            recurrent_state = rs.last_recurrent_state if rs.last_recurrent_state is not None else \
                torch.zeros(
                    (bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                    dtype = torch.float,
                    device = self.device
                )

            save_state = True
        else:
            conv_state = None
            recurrent_state = None
            save_state = False

        # C++ path (temporarily disabled for TP debugging)
        if False and self.bc is not None and bsz == 1 and seqlen == 1 and save_state:
            y = torch.empty_like(x)
            mixed_qkv = self.bc.run_bsz1_a(x)
            mixed_qkv = causal_conv1d_update_function(
                mixed_qkv,
                conv_state,  # Updated inplace
                self.conv1d_weight.squeeze(1),
                self.conv1d_bias,
            )
            self.bc.run_bsz1_b(mixed_qkv, y, recurrent_state)
            x = y

        # Torch path
        else:
            # Projections
            #
            # NOTE:
            # Qwen3.5 uses split projections (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a),
            # while Qwen3-Next uses fused projections. The fused C++ helper expects the
            # packed layout used by fused projections; applying it to split qkv tensors
            # causes incorrect head ordering and broken generations.
            if self.qkvz_proj is not None and self.ba_proj is not None:
                if not hasattr(self, "_fused_path_diag"):
                    import sys
                    print(f"[GDN] Using FUSED path (qkvz_proj), layer={self.layer_idx}, tp_reduce={self.tp_reduce}", file=sys.stderr, flush=True)
                    self._fused_path_diag = True

                qkvz = self.qkvz_proj.forward(x, params)
                ba = self.ba_proj.forward(x, params)

                if not hasattr(self, "_qkvz_diag") and self.tp_reduce:
                    import sys
                    qkvz_norm = qkvz.float().norm().item()
                    qkvz_proj_out_features = self.qkvz_proj.out_features if hasattr(self.qkvz_proj, 'out_features') else None
                    print(f"[GDN qkvz] layer={self.layer_idx} qkvz: norm={qkvz_norm:.4f}, shape={qkvz.shape}, qkvz_proj.out_features={qkvz_proj_out_features}", file=sys.stderr, flush=True)
                    self._qkvz_diag = True

                mixed_qkv = torch.empty((bsz, self.fdim_qkv, seqlen), dtype = torch.bfloat16, device = self.device)
                z = torch.empty((bsz, seqlen, self.num_v_heads, self.v_head_dim), dtype = torch.bfloat16, device = self.device)
                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

                ext.gated_delta_net_fused_op(
                    qkvz, ba,
                    self.dt_bias,
                    self.a_log,
                    mixed_qkv, z, beta, g,
                    self.num_k_heads,
                    self.num_v_heads,
                    self.k_head_dim,
                    self.v_head_dim
                )
            else:
                # TODO: Bound class and/or graph for this part
                if not hasattr(self, "_split_path_diag"):
                    import sys
                    print(f"[GDN] Using SPLIT path (z_proj), layer={self.layer_idx}, tp_reduce={self.tp_reduce}, params tp_reduce={params.get('tp_reduce')}", file=sys.stderr, flush=True)
                    self._split_path_diag = True

                qkv = self.qkv_proj.forward(x, params)
                z_raw = self.z_proj.forward(x, params)

                if not hasattr(self, "_z_raw_diag") and self.tp_reduce:
                    import sys
                    z_raw_norm = z_raw.float().norm().item()
                    print(f"[GDN z_raw] layer={self.layer_idx} z_raw (before view): norm={z_raw_norm:.4f}, shape={z_raw.shape}", file=sys.stderr, flush=True)
                    self._z_raw_diag = True

                z = z_raw.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)
                b = self.b_proj.forward(x, params)
                a = self.a_proj.forward(x, params)

                mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()

                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

                ext.gated_delta_net_fused_op_2(
                    b, a,
                    self.dt_bias,
                    self.a_log,
                    beta, g
                )

            # Convolution
            # TODO: Figure out an alternative or write a new kernel that won't require transposing qkv back and forth
            if conv_state is None:
                if save_state:
                    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                    rs.last_conv_state = conv_state
                mixed_qkv = causal_conv1d_fwd_function(
                    mixed_qkv,
                    self.conv1d_weight.squeeze(1),
                    self.conv1d_bias,
                )
            else:
                mixed_qkv = causal_conv1d_update_function(
                    mixed_qkv,
                    conv_state,  # Updated inplace
                    self.conv1d_weight.squeeze(1),
                    self.conv1d_bias,
                )

            # Use chunked rule when advantageous and available
            # TODO: At least warn if chunked rule (i.e. flash-linear-attention) is not available
            #       since performance will tank on prompt ingestion
            if seqlen >= self.num_v_heads and chunk_gated_delta_rule is not None:
                mixed_qkv = mixed_qkv.transpose(1, 2)

                q, k, v = torch.split(mixed_qkv, [self.k_dim, self.k_dim, self.v_dim], dim = -1)
                q = q.view(bsz, seqlen, -1, self.k_head_dim)
                k = k.view(bsz, seqlen, -1, self.k_head_dim)
                v = v.view(bsz, seqlen, -1, self.v_head_dim)

                # Grouped attn
                if self.num_v_heads // self.num_k_heads > 1:
                    q = q.repeat_interleave(self.num_v_groups, dim = 2)
                    k = k.repeat_interleave(self.num_v_groups, dim = 2)

                core_attn_out, recurrent_state = chunk_gated_delta_rule(
                    q, k, v,
                    g = g,
                    beta = beta,
                    initial_state = recurrent_state,
                    output_final_state = save_state,
                    use_qk_l2norm_in_kernel = True,
                )

            else:
                core_attn_out = torch.empty(
                    (bsz, seqlen, self.num_v_heads, self.v_head_dim),
                    dtype = torch.bfloat16,
                    device = self.device,
                )

                mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()
                if recurrent_state is None:
                    recurrent_state = torch.zeros(
                        (bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                        dtype = torch.float,
                        device = self.device
                    )
                ext.cuda_recurrent_gated_delta_rule(
                    mixed_qkv,
                    g,
                    beta,
                    recurrent_state,
                    core_attn_out,
                    self.num_k_heads,
                    self.num_v_heads,
                    self.k_head_dim,
                    self.v_head_dim
                )

            # Norm - per-head normalization (weight is [v_head_dim] not [num_v_heads * v_head_dim])
            if not hasattr(self, "_gate_diag") and self.layer_idx == 0 and not params.get("prefill") and self.tp_reduce:
                import sys
                z_norm = z.float().norm().item()
                core_norm = core_attn_out.float().norm().item()
                z_proj_out_features = self.z_proj.out_features if hasattr(self.z_proj, 'out_features') else None
                z_proj_in_features = self.z_proj.in_features if hasattr(self.z_proj, 'in_features') else None
                print(f"[GDN gate] layer={self.layer_idx} z: norm={z_norm:.4f}, core_attn_out: norm={core_norm:.4f}, z_proj out/in features={z_proj_out_features}/{z_proj_in_features}", file=sys.stderr, flush=True)
                self._gate_diag = True
            core_attn_out = self.norm.forward(core_attn_out, params, gate=z)

            core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)

            if not hasattr(self, "_pre_o_proj_diag") and self.layer_idx == 0 and not params.get("prefill") and self.tp_reduce:
                import sys
                pre_o_norm = core_attn_out.float().norm().item()
                print(f"[GDN pre_o_proj] layer={self.layer_idx} core_attn_out: norm={pre_o_norm:.4f}, shape={core_attn_out.shape}", file=sys.stderr, flush=True)
                self._pre_o_proj_diag = True

            # Output projection
            x = self.o_proj.forward(core_attn_out, params)

            if not hasattr(self, "_o_proj_diag") and self.layer_idx == 0 and not params.get("prefill") and self.tp_reduce:
                import sys
                pre_reduce_norm = x.float().norm().item()
                print(f"[GDN o_proj] layer={self.layer_idx} pre_all_reduce: norm={pre_reduce_norm:.4f}", file=sys.stderr, flush=True)
                self._o_proj_diag = True

        # Update cache
        if save_state:
            rs.last_recurrent_state = recurrent_state
            rs.last_conv_state = conv_state
            if not rs.batched:
                rs.position += seqlen
            else:
                rs.positions = [r + seqlen for r in rs.positions]

        # TP reduction: o_proj is input-split, so we need all_reduce to sum partial outputs
        # No scaling needed since we disabled normalization in TP mode
        if self.tp_reduce:
            params["backend"].all_reduce(x)

        return to2(x, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.a_log, self.key_a_log),
            (self.dt_bias, self.key_dt_bias),
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
        ]:
            if x is not None:
                t[k] = x
        return t


    def new_recurrent_state(self):
        return GDN_RecurrentState()


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:

        storage = 0
        if self.qkvz_proj is not None:
            storage += self.qkvz_proj.storage_size()
        if self.qkv_proj is not None:
            storage += self.qkv_proj.storage_size()
        if self.z_proj is not None:
            storage += self.z_proj.storage_size()
        if self.ba_proj is not None:
            storage += self.ba_proj.storage_size()
        if self.b_proj is not None:
            storage += self.b_proj.storage_size()
        if self.a_proj is not None:
            storage += self.a_proj.storage_size()
        storage += self.o_proj.storage_size()

        overhead_d = self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        overhead_s += self.k_dim * torch.bfloat16.itemsize  # q
        overhead_s += self.k_dim * torch.bfloat16.itemsize  # k
        overhead_s += self.v_dim * torch.bfloat16.itemsize  # v
        overhead_s += self.v_dim * torch.bfloat16.itemsize  # z
        overhead_s += self.num_v_heads * torch.bfloat16.itemsize  # beta
        overhead_s += self.num_v_heads * torch.float.itemsize  # g
        overhead_s += self.num_v_heads * self.v_head_dim * torch.bfloat16.itemsize  # core_attn_out

        recons_list = [self.o_proj.recons_size()]
        if self.qkvz_proj is not None:
            recons_list.append(self.qkvz_proj.recons_size())
        if self.qkv_proj is not None:
            recons_list.append(self.qkv_proj.recons_size())
        if self.z_proj is not None:
            recons_list.append(self.z_proj.recons_size())
        recons = max(recons_list)

        channel_width = 1
        channels_to_split = self.num_k_heads
        while channel_width * self.k_head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.k_head_dim) % 128 == 0, \
            "Model's K heads cannot divide into 128-channel tensors"

        tpa = TPAllocation(
            key = self.key,
            channel_width = channel_width,
            channel_unit = "heads",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = channels_to_split,
            limit_key = "attn"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": GatedDeltaNet,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "k_head_dim": self.k_head_dim,
                "v_head_dim": self.v_head_dim,
                "rms_norm_eps": self.rms_norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "out_dtype": self.out_dtype,
            },
            "num_k_heads": self.num_k_heads,
            "num_v_heads": self.num_v_heads,
            "num_v_groups": self.num_v_groups,
            **{name: _export(getattr(self, name, None)) for name in (
                "qkvz_proj",
                "qkv_proj",
                "z_proj",
                "ba_proj",
                "b_proj",
                "a_proj",
                "o_proj",
                "norm",
            )},
            "a_log": producer.send(self.a_log),
            "dt_bias": producer.send(self.dt_bias),
            "conv1d_weight": producer.send(self.conv1d_weight),
            "conv1d_bias": producer.send(self.conv1d_bias),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        import sys
        device = local_context["device"]
        consumer = local_context["consumer"]

        key = exported["kwargs"]["key"]
        k_head_dim = exported["kwargs"]["k_head_dim"]
        v_head_dim = exported["kwargs"]["v_head_dim"]
        hidden_size = exported["kwargs"]["hidden_size"]
        num_v_groups = exported["num_v_groups"]
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "heads"

        num_k_heads = last - first
        num_v_heads = num_k_heads * num_v_groups
        num_k_heads_full = len(local_context["active_devices"]) * num_k_heads

        if not hasattr(sys.modules[__name__], "_gdn_tp_import_diag"):
            print(f"[GDN tp_import] first={first}, last={last}, num_k_heads={num_k_heads}, num_v_groups={num_v_groups}, num_v_heads={num_v_heads}", file=sys.stderr, flush=True)
            sys.modules[__name__]._gdn_tp_import_diag = True

        k_dim = k_head_dim * num_k_heads
        v_dim = v_head_dim * num_v_heads

        fdim_qkvz = 2 * k_dim + 2 * v_dim
        fdim_qkv = 2 * k_dim + v_dim

        qkvz_split = (True, first * (2 * k_head_dim + 2 * v_head_dim * num_v_groups),
                      last * (2 * k_head_dim + 2 * v_head_dim * num_v_groups)) \
            if num_k_heads else None
        qkv_split = (True, first * (2 * k_head_dim + v_head_dim * num_v_groups),
                     last * (2 * k_head_dim + v_head_dim * num_v_groups)) \
            if num_k_heads else None
        z_split = (True, first * v_head_dim * num_v_groups, last * v_head_dim * num_v_groups) \
            if num_k_heads else None
        ba_split = (True, first * 2 * num_v_groups, last * 2 * num_v_groups) \
            if num_k_heads else None
        b_split = (True, first * num_v_groups, last * num_v_groups) \
            if num_k_heads else None
        a_split = (True, first * num_v_groups, last * num_v_groups) \
            if num_k_heads else None
        # o_proj is input-split: each device gets a slice of the input dimension (rows of weight matrix)
        # matching its partial head outputs (num_v_heads * v_head_dim per device)
        o_split = (False, first * v_head_dim * num_v_groups, last * v_head_dim * num_v_groups) \
            if num_k_heads else None
        # Conv operates on mixed_qkv which has fdim_qkv channels per head group
        conv_split = (first * (2 * k_head_dim + v_head_dim * num_v_groups),
                      last * (2 * k_head_dim + v_head_dim * num_v_groups)) \
            if num_k_heads else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        # Explicitly construct with split head counts
        kwargs = exported["kwargs"]

        module = GatedDeltaNet(
            config = None,
            key = kwargs["key"],
            layer_idx = kwargs["layer_idx"],
            hidden_size = kwargs["hidden_size"],
            k_head_dim = kwargs["k_head_dim"],
            v_head_dim = kwargs["v_head_dim"],
            num_k_heads = num_k_heads,
            num_v_heads = num_v_heads,
            rms_norm_eps = kwargs["rms_norm_eps"],
            conv_kernel_size = kwargs["conv_kernel_size"],
            out_dtype = kwargs.get("out_dtype"),
            qkvz_proj = _import_split("qkvz_proj", qkvz_split),
            qkv_proj = _import_split("qkv_proj", qkv_split),
            z_proj = _import_split("z_proj", z_split),
            ba_proj = _import_split("ba_proj", ba_split),
            b_proj = _import_split("b_proj", b_split),
            a_proj = _import_split("a_proj", a_split),
            o_proj = _import_split("o_proj", o_split),
            norm = _import("norm"),
        )

        module.device = device

        # Split dt_bias and a_log according to value heads
        dt_bias_full = consumer.recv(exported["dt_bias"], cuda = True)
        a_log_full = consumer.recv(exported["a_log"], cuda = True)

        v_head_split = (first * num_v_groups, last * num_v_groups) if num_k_heads else None

        if v_head_split and dt_bias_full is not None:
            module.dt_bias = dt_bias_full[v_head_split[0]:v_head_split[1]].contiguous()
        else:
            module.dt_bias = dt_bias_full

        if v_head_split and a_log_full is not None:
            module.a_log = a_log_full[v_head_split[0]:v_head_split[1]].contiguous()
        else:
            module.a_log = a_log_full

        conv1d_weight = consumer.recv(exported["conv1d_weight"], cuda = True)
        if conv_split and conv1d_weight is not None:
            module.conv1d_weight = conv1d_weight[conv_split[0]:conv_split[1], :].contiguous()
        else:
            module.conv1d_weight = conv1d_weight

        conv1d_bias = consumer.recv(exported["conv1d_bias"], cuda = True)
        if conv_split and conv1d_bias is not None:
            module.conv1d_bias = conv1d_bias[conv_split[0]:conv_split[1]].contiguous()
        else:
            module.conv1d_bias = conv1d_bias

        # GDN uses input-split o_proj, so all_reduce is needed to sum partial outputs
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)

        torch.cuda.synchronize()
        return module
