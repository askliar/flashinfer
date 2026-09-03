# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Jointly tune the Triton causal Conv1d and FlashInfer ReplaySSM SSU.

This benchmark keeps co-tuning outside the production API.  It uses the
framework-neutral Triton Conv1d reference as the upstream PDL producer and
combines each Conv1d launch configuration with every SSU tactic exposed by
``CheckpointingSSURunner``.  FlashInfer's generic autotuner then measures the
complete CUDA-graph chain and selects one paired tactic.

Example::

    python benchmarks/bench_replayssm_conv1d_cotune.py \
        --batches 1,2,4,8,16 --mtp-length 6 --output cotune.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "mamba"))

import bench_checkpointing_ssu as bench  # noqa: E402
from flashinfer.autotuner import (  # noqa: E402
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from flashinfer.mamba.checkpointing_ssu import (  # noqa: E402
    _ALGORITHM_AUTO,
    _get_checkpointing_ssu_runner,
)
from triton_reference.causal_conv1d_triton import (  # noqa: E402
    causal_conv1d_update,
)


ConvConfig = tuple[int, int, int]
SSUTactic = tuple[int, int, int, int]
PairedTactic = tuple[int, int, int, int, int, int, int]


def _parse_conv_config(value: str) -> ConvConfig:
    fields = tuple(int(field) for field in value.split(":"))
    if len(fields) != 3 or min(fields) <= 0:
        raise argparse.ArgumentTypeError(
            f"expected positive BLOCK_N:WARPS:STAGES, got {value!r}"
        )
    return fields


def _ssu_inputs(inputs: list[Any]) -> list[Any]:
    return inputs[:22]


def _ssu_runner(inputs: list[Any], enable_pdl: bool):
    state = inputs[0]  # state
    x = inputs[1]  # x
    dt = inputs[2]  # dt
    A = inputs[3]  # A
    B = inputs[4]  # B
    x_cache = inputs[7]  # x_cache
    D = inputs[12]  # D
    dt_bias = inputs[14]  # dt_bias
    state_batch_indices = inputs[15]  # state_batch_indices
    state_scale = inputs[16]  # state_scale
    nheads = state.size(1)
    ngroups = B.size(-2)
    heads_per_group = nheads // ngroups
    npredicted = x.size(1)
    max_window = x_cache.size(2) - npredicted
    weight_dtype = D.dtype if D is not None else dt_bias.dtype
    module_base_args = (
        state.dtype,
        x.dtype,
        dt.dtype,
        weight_dtype,
        A.dtype,
        state_batch_indices.dtype
        if state_batch_indices is not None
        else torch.int32,
        state_scale.dtype if state_scale is not None else None,
        state.size(2),
        state.size(3),
        npredicted,
        max_window,
        heads_per_group,
        ngroups,
        0,  # philox_rounds
        enable_pdl,
    )
    optional_tensor_presence = tuple(
        inputs[index] is not None for index in range(12, 19)
    )
    return _get_checkpointing_ssu_runner(
        module_base_args,
        True,  # dt_softplus
        -1,  # pad_slot_id
        _ALGORITHM_AUTO,
        0,  # requested_d_split
        0,  # precompute_heads_per_cta
        heads_per_group,
        optional_tensor_presence,
    )


class PairedRunner(TunableRunner):
    """Apply one Conv1d launch configuration and one SSU tactic."""

    def __init__(
        self,
        conv_configs: tuple[ConvConfig, ...],
        *,
        enable_pdl: bool,
    ) -> None:
        self.conv_configs = conv_configs
        self.enable_pdl = enable_pdl

    def get_valid_tactics(
        self, inputs: list[Any], profile: OptimizationProfile
    ) -> list[PairedTactic]:
        runner = _ssu_runner(inputs, self.enable_pdl)
        ssu_tactics = runner.get_valid_tactics(_ssu_inputs(inputs), profile)
        return [
            (*conv_config, *ssu_tactic)
            for conv_config in self.conv_configs
            for ssu_tactic in ssu_tactics
        ]

    def get_cache_key_extras(self, inputs: list[Any]) -> tuple[Any, ...]:
        del inputs
        return (self.conv_configs, self.enable_pdl)

    def forward(
        self,
        inputs: list[Any],
        tactic: PairedTactic | int = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        runner = _ssu_runner(inputs, self.enable_pdl)
        if do_preparation:
            runner(_ssu_inputs(inputs), tactic=-1, do_preparation=True)
            return
        if tactic == -1:
            conv_config = self.conv_configs[0]
            ssu_tactic: SSUTactic | int = -1
        elif isinstance(tactic, tuple) and len(tactic) == 7:
            conv_config = tactic[:3]
            ssu_tactic = tactic[3:]
        else:
            raise ValueError(f"invalid paired tactic: {tactic}")

        block_n, conv_warps, conv_stages = conv_config
        conv_out = causal_conv1d_update(
            inputs[22].transpose(1, 2),  # xbc_input
            inputs[23].transpose(1, 2),  # conv_state
            inputs[24],  # conv_weight
            inputs[25],  # conv_bias
            activation="silu",
            launch_dependent_kernels=self.enable_pdl,
            _block_n=block_n,
            _num_warps=conv_warps,
            _num_stages=conv_stages,
        )
        x_shape = inputs[1].shape  # x
        B_shape = inputs[4].shape  # B
        conv_flat = conv_out.transpose(1, 2).reshape(x_shape[0] * x_shape[1], -1)
        group_width = B_shape[2] * B_shape[3]
        x_flat, B_flat, C_flat = torch.split(
            conv_flat,
            (x_shape[2] * x_shape[3], group_width, group_width),
            dim=-1,
        )
        call_inputs = _ssu_inputs(inputs)
        call_inputs[1] = x_flat.view(x_shape)  # x
        call_inputs[4] = B_flat.view(B_shape)  # B
        call_inputs[5] = C_flat.view_as(inputs[5])  # C
        runner(call_inputs, tactic=ssu_tactic)


def _profile_inputs(inputs: bench.KernelInputs) -> list[Any]:
    return [
        inputs.state_work,  # 0: state
        inputs.x,  # 1: x shape template
        inputs.dt,  # 2: dt
        inputs.A,  # 3: A
        inputs.B,  # 4: B shape template
        inputs.C,  # 5: C shape template
        inputs.out_incr,  # 6: out
        inputs.x_cache_work,  # 7: x_cache
        inputs.B_cache_work,  # 8: B_cache
        inputs.dt_cache_work,  # 9: dt_cache
        inputs.ring_start,  # 10: ring_start
        inputs.prev_tokens_i32,  # 11: prev_num_accepted_tokens
        inputs.D,  # 12: D
        None,  # 13: z
        inputs.dt_bias,  # 14: dt_bias
        None,  # 15: state_batch_indices
        inputs.state_scale_work,  # 16: state_scale
        None,  # 17: rand_seed
        None,  # 18: cu_seqlens
        inputs.cb_scaled,  # 19: cb_scaled
        inputs.cumAdt_vec,  # 20: cumAdt_vec
        inputs.cb_old,  # 21: cb_old
        inputs.xbc_input_work.transpose(1, 2),  # 22: xbc_input backing
        inputs.conv_state_work.transpose(1, 2),  # 23: conv_state backing
        inputs.conv_weight,  # 24: conv_weight
        inputs.conv_bias,  # 25: conv_bias
    ]


def _set_feature_contiguous_conv_state(inputs: bench.KernelInputs) -> None:
    backing = torch.randn(
        inputs.batch,
        3,
        inputs.conv_dim,
        device="cuda",
        dtype=inputs.x.dtype,
    )
    inputs.conv_state0 = backing.transpose(1, 2)
    inputs.conv_state_work = backing.clone().transpose(1, 2)


def _tune(
    inputs: bench.KernelInputs,
    conv_configs: tuple[ConvConfig, ...],
    *,
    enable_pdl: bool,
    warmup: int,
    repeat: int,
) -> tuple[PairedRunner, PairedTactic]:
    runner = PairedRunner(conv_configs, enable_pdl=enable_pdl)
    tuner = AutoTuner(warmup=warmup, repeat=repeat)
    tuner.is_tuning_mode = True
    _, tactic = tuner.choose_one(
        "causal_conv1d_checkpointing_ssu",
        [runner],
        TuningConfig(
            use_cuda_graph=True,
            use_cold_l2_cache=True,
            profile_arena_input_indices=(
                0,
                1,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                19,
                20,
                21,
                23,
                24,
                25,
            ),
        ),
        _profile_inputs(inputs),
    )
    if not isinstance(tactic, tuple) or len(tactic) != 7:
        raise RuntimeError(f"autotuner did not select a paired tactic: {tactic}")
    return runner, tactic


def _measure(
    inputs: bench.KernelInputs,
    run: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    tag: str,
) -> tuple[float, float, float]:
    return bench._time_kernel(
        bench.TimingOptions(
            warmup=warmup,
            iters=iterations,
            cupti=False,
            cuda_graph=True,
            l2_flush=False,
        ),
        run,
        inputs.reset_with_conv1d,
        tag,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--mtp-length", type=int, default=6)
    parser.add_argument("--max-window", type=int, default=16)
    parser.add_argument("--nheads", type=int, default=64)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument(
        "--conv-configs",
        type=_parse_conv_config,
        nargs="+",
        default=[
            (block_n, warps, stages)
            for block_n in (64, 128, 256)
            for warps in (4, 8)
            for stages in (1, 2, 3)
        ],
    )
    parser.add_argument(
        "--fixed-conv", type=_parse_conv_config, default=(128, 4, 3)
    )
    parser.add_argument("--pdl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-warmup", type=int, default=2)
    parser.add_argument("--tune-repeat", type=int, default=10)
    parser.add_argument("--measure-warmup", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.pdl and torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("--pdl requires an SM90-or-newer GPU")
    if not 0 < args.mtp_length <= args.max_window:
        raise ValueError("--mtp-length must be in [1, --max-window]")
    conv_configs = tuple(args.conv_configs)
    fixed_conv_configs = (args.fixed_conv,)
    results = []
    bench._init_l2_flush()

    for batch in (int(value) for value in args.batches.split(",")):
        inputs = bench.build_kernel_inputs(
            batch=batch,
            mtp_len=args.mtp_length,
            max_window=args.max_window,
            state_dtype=torch.float32,
            act_dtype=torch.bfloat16,
            nheads=args.nheads,
            head_dim=args.head_dim,
            d_state=args.d_state,
            ngroups=args.ngroups,
            two_kernel=True,
        )
        _set_feature_contiguous_conv_state(inputs)
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        inputs.prev_tokens_i32.copy_((slots * args.max_window) % (args.max_window + 1))

        fixed_runner, fixed_tactic = _tune(
            inputs,
            fixed_conv_configs,
            enable_pdl=args.pdl,
            warmup=args.tune_warmup,
            repeat=args.tune_repeat,
        )
        paired_runner, paired_tactic = _tune(
            inputs,
            conv_configs,
            enable_pdl=args.pdl,
            warmup=args.tune_warmup,
            repeat=args.tune_repeat,
        )
        profile_inputs = _profile_inputs(inputs)
        fixed_timing = _measure(
            inputs,
            lambda: fixed_runner(profile_inputs, tactic=fixed_tactic),
            warmup=args.measure_warmup,
            iterations=args.measure_iters,
            tag=f"fixed_b{batch}_{fixed_tactic}",
        )
        paired_timing = _measure(
            inputs,
            lambda: paired_runner(profile_inputs, tactic=paired_tactic),
            warmup=args.measure_warmup,
            iterations=args.measure_iters,
            tag=f"paired_b{batch}_{paired_tactic}",
        )
        result = {
            "batch": batch,
            "fixed_tactic": fixed_tactic,
            "paired_tactic": paired_tactic,
            "fixed_median_us": fixed_timing[0],
            "paired_median_us": paired_timing[0],
            "paired_speedup": fixed_timing[0] / paired_timing[0],
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
