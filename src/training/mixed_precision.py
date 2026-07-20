# src/training/mixed_precision.py
"""
Automatic Mixed Precision (AMP) manager.

Supports:
- FP16 mixed precision with gradient scaling
- BF16 mixed precision (no scaling needed)
- Full FP32 fallback
- Gradient clipping with automatic unscaling
- Checkpoint save/load for training resumption

Designed for research workflows where precision mode
is controlled via configuration.
"""

from __future__ import annotations

from typing import Optional
from contextlib import nullcontext
import logging

import torch
import torch.cuda.amp as amp

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """
    Manages Automatic Mixed Precision training.

    Handles:
        - autocast context selection (FP16 / BF16 / FP32)
        - Gradient scaling (FP16 only)
        - Gradient unscaling before clipping
        - State serialization for checkpointing

    Parameters
    ----------
    enabled : bool
        Whether to enable AMP.

    device : str
        Device string (AMP only works on CUDA).

    use_bfloat16 : bool
        Use BF16 instead of FP16. Requires Ampere+ GPU.
        BF16 has more dynamic range and needs no gradient scaling.

    scale_window : int
        Window for dynamic growth of the gradient scale factor.
    """

    def __init__(
        self,
        enabled: bool = True,
        device: str = "cuda",
        use_bfloat16: bool = False,
        scale_window: int = 2000,
    ) -> None:
        self._enabled = enabled and device == "cuda" and torch.cuda.is_available()
        self._device = device
        self._use_bfloat16 = use_bfloat16
        self._scale_window = scale_window

        if not self._enabled:
            self.dtype = torch.float32
            self.scaler = None
            logger.info("AMP disabled – using FP32")
            return

        if use_bfloat16 and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.scaler = None
            logger.info("AMP enabled – BF16 (no gradient scaling)")
        else:
            self.dtype = torch.float16
            self.scaler = amp.GradScaler(
                init_scale=2.0**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=scale_window,
                enabled=True,
            )
            logger.info(
                "AMP enabled – FP16 with gradient scaling "
                "(window=%d steps)", scale_window
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether AMP is active."""
        return self._enabled

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def autocast(self):
        """
        Context manager for automatic dtype selection.

        Usage
        -----
        with amp_manager.autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)
        """
        if self._enabled:
            return amp.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for backward pass.

        Returns
        -------
        Scaled loss tensor (or original if no scaler).
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform an optimizer step (AMP‑aware)."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update gradient scaler after each optimizer step."""
        if self.scaler is not None:
            self.scaler.update()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients before clipping.

        Must be called before `torch.nn.utils.clip_grad_norm_`
        when using FP16 AMP.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_scaling_factor(self) -> float:
        """Return current gradient scale (for logging)."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return scaler state for serialization."""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scaler state from checkpoint."""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)
