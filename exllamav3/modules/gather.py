from __future__ import annotations
from typing_extensions import override
from . import Module
import torch

class OutputGather(Module):
    def __init__(
        self,
        config: None,
        key: str,
        device: int,
        output_device: int,
        gather_devices: list[int],
        ldims: list[int],
    ):
        super().__init__(config, key, None)
        self.device = device
        self.output_device = output_device
        self.gather_devices = gather_devices
        self.ldims = ldims
        self.odim = sum(ldims)
        self.active = device == output_device or device in gather_devices

    @override
    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.Device, **kwargs):
        raise NotImplementedError()

    @override
    def unload(self):
        raise NotImplementedError()

    @override
    def get_tensors(self):
        raise NotImplementedError()

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:

        if len(self.gather_devices) == 1 and self.device == self.output_device:
            return x

        if not self.active:
            return None

        backend = params["backend"]

        if self.output_device == self.device:
            out_shape = list(x.shape)
            out_shape[-1] = self.odim
            out_tensor = torch.empty(*out_shape, dtype = x.dtype, device = x.device)
        else:
            out_tensor = None

        # print(f"Gather:  device {self.device}, ldims {self.ldims}")

        # One-shot diagnostic
        import sys
        if not hasattr(OutputGather, "_diag_done"):
            OutputGather._diag_done = False
        if not OutputGather._diag_done and self.output_device == self.device:
            OutputGather._diag_done = True
            print(f"[Gather diag] dev={self.device} gather_devices={self.gather_devices} ldims={self.ldims} odim={self.odim} local_shape={tuple(x.shape)}", file=sys.stderr, flush=True)
            # Find our offset in the gathered tensor
            our_offset = 0
            our_ldim = x.shape[-1]
            for gd, ld in zip(self.gather_devices, self.ldims):
                if gd == self.device:
                    our_ldim = ld
                    break
                our_offset += ld
            print(f"[Gather diag] our_offset={our_offset} our_ldim={our_ldim}", file=sys.stderr, flush=True)
            # Save pre-gather local tensor for verification
            x_local_copy = x.clone()

        backend.gather(x, out_tensor, self.gather_devices, self.output_device, self.ldims)

        # Verify gather placed our local data at the right position
        if hasattr(OutputGather, '_diag_done') and OutputGather._diag_done and self.output_device == self.device:
            if not hasattr(OutputGather, '_diag_verified'):
                OutputGather._diag_verified = True
                gathered_slice = out_tensor[..., our_offset:our_offset + our_ldim]
                match = torch.allclose(x_local_copy.float(), gathered_slice.float(), atol=1e-4)
                print(f"[Gather diag] local_slice_match={match} local_sum={x_local_copy.sum().item():.4f} gathered_slice_sum={gathered_slice.sum().item():.4f}", file=sys.stderr, flush=True)
                # Also print the argmax token from gathered vs what it would be from just local
                full_argmax = out_tensor[0, -1].float().argmax().item()
                local_argmax = x_local_copy[0, -1].float().argmax().item()
                print(f"[Gather diag] full_argmax={full_argmax} local_argmax_in_local={local_argmax} local_argmax_in_global={local_argmax + our_offset}", file=sys.stderr, flush=True)

        return out_tensor
