from collections.abc import Sequence
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        device: str = "cpu",
        is_pytorch: bool = False,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._device = device
        self._is_pytorch_model = is_pytorch

        if self._is_pytorch_model:
            self._model = self._model.to(device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        if not self._is_pytorch_model:
            self._rng, sample_rng = jax.random.split(self._rng)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            if self._is_pytorch_model:
                import torch
                noise = torch.from_numpy(noise)
                noise = noise.to(self._device)
            else:
                noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        actions = (
            self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **sample_kwargs)
            if not self._is_pytorch_model
            else self._sample_actions(inputs, **sample_kwargs)
        )
        outputs = {"state": inputs["state"], "actions": actions}
        # Unbatch and convert to np.ndarray.
        if self._is_pytorch_model:
            # For PyTorch models, handle CUDA tensors by moving to CPU first
            def convert_pytorch_tensor(x):
                if hasattr(x, 'cpu'):  # PyTorch tensor
                    return np.asarray(x[0, ...].detach().cpu())
                else:
                    return np.asarray(x[0, ...])
            outputs = jax.tree.map(convert_pytorch_tensor, outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs
