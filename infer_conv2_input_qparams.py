from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BR0_CH = 13
BR1_CH = 13
BR2_CH = 14
CONCAT_CH = BR0_CH + BR1_CH + BR2_CH
IN_DIM = 15
TIME_DIM = 410

BRANCH_SCALES = np.array(
    [0.027724377811, 0.035352136940, 0.035856030881],
    dtype=np.float64,
)
BRANCH_ZPS = np.array([6.0, 5.0, 5.0], dtype=np.float64)
CONV2_OUTPUT_SCALE = 0.059495572001
CONV2_OUTPUT_ZP = 132


@dataclass(order=True)
class SearchResult:
    mismatch: int
    abs_error: int
    max_error: int
    scale: float
    zp: int


def load_numeric_text(path: Path, is_float: bool = False) -> np.ndarray:
    tokens = []
    for token in path.read_text().replace("{", ",").replace("}", ",").replace("\n", ",").split(","):
        token = token.strip()
        if token:
            tokens.append(float(token) if is_float else int(token))
    dtype = np.float64 if is_float else np.int32
    return np.array(tokens, dtype=dtype)


def load_ref_bundle(ref_root: Path) -> dict[str, np.ndarray | float | int]:
    trace_root = ref_root / "trace"
    conv_root = ref_root / "conv_spatial"

    br0 = load_numeric_text(trace_root / "trace_ms_conv1_branches_0_3.txt").reshape(BR0_CH, IN_DIM, TIME_DIM)
    br1 = load_numeric_text(trace_root / "trace_ms_conv1_branches_1_3.txt").reshape(BR1_CH, IN_DIM, TIME_DIM)
    br2 = load_numeric_text(trace_root / "trace_ms_conv1_branches_2_3.txt").reshape(BR2_CH, IN_DIM, TIME_DIM)
    conv2_ref = load_numeric_text(trace_root / "trace_conv2.txt").reshape(CONCAT_CH, TIME_DIM)
    weights = load_numeric_text(conv_root / "conv2_weight_int8.txt").reshape(CONCAT_CH, CONCAT_CH, IN_DIM)
    bias = load_numeric_text(conv_root / "conv2_bias_int32.txt")
    weight_scale = float(load_numeric_text(conv_root / "conv2_weight_scales.txt", is_float=True)[0])

    exported_scale = None
    exported_zp = None
    for line in (conv_root / "conv2_meta.txt").read_text().splitlines():
        if line.startswith("Input_Scale:"):
            exported_scale = float(line.split(":", 1)[1].strip())
        elif line.startswith("Input_ZP:"):
            exported_zp = int(line.split(":", 1)[1].strip())

    x = np.concatenate([br0, br1, br2], axis=0).astype(np.float64)
    x_scale = np.concatenate(
        [
            np.full(BR0_CH, BRANCH_SCALES[0]),
            np.full(BR1_CH, BRANCH_SCALES[1]),
            np.full(BR2_CH, BRANCH_SCALES[2]),
        ]
    )[:, None, None]
    x_zp = np.concatenate(
        [
            np.full(BR0_CH, BRANCH_ZPS[0]),
            np.full(BR1_CH, BRANCH_ZPS[1]),
            np.full(BR2_CH, BRANCH_ZPS[2]),
        ]
    )[:, None, None]

    return {
        "real_inputs": (x - x_zp) * x_scale,
        "conv2_ref": conv2_ref.astype(np.int16),
        "weights": weights.astype(np.int16),
        "bias": bias.astype(np.int32),
        "weight_scale": weight_scale,
        "exported_scale": exported_scale,
        "exported_zp": exported_zp,
    }


def quantize_to_concat_domain(real_inputs: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = np.rint(real_inputs / scale + zp)
    q = np.clip(q, 0, 255)
    return q.astype(np.int16) - zp


def conv2_inner_sum(weights: np.ndarray, centered_inputs: np.ndarray) -> np.ndarray:
    return np.einsum("oir,irt->ot", weights, centered_inputs, optimize=True)


def score_with_fixed_bias(
    inner_sum: np.ndarray,
    bias: np.ndarray,
    ref: np.ndarray,
    scale: float,
    weight_scale: float,
) -> tuple[int, int, int]:
    requant = scale * weight_scale / CONV2_OUTPUT_SCALE
    out = np.rint((inner_sum + bias[:, None]) * requant + CONV2_OUTPUT_ZP)
    out = np.clip(out, 0, 255).astype(np.int16)
    diff = out - ref
    return int(np.count_nonzero(diff)), int(np.abs(diff).sum()), int(np.abs(diff).max())


def score_with_refit_bias(
    inner_sum: np.ndarray,
    ref: np.ndarray,
    scale: float,
    weight_scale: float,
    bias_search_radius: int,
) -> tuple[int, int, int]:
    requant = scale * weight_scale / CONV2_OUTPUT_SCALE
    # 用中位数给每个输出通道一个初值，再局部搜索最优整数 bias。
    target_acc = (ref.astype(np.float64) - CONV2_OUTPUT_ZP) / requant
    bias_seed = np.rint(np.median(target_acc - inner_sum, axis=1)).astype(np.int32)

    total_mismatch = 0
    total_abs_error = 0
    max_error = 0
    for oc in range(CONCAT_CH):
        candidates = bias_seed[oc] + np.arange(-bias_search_radius, bias_search_radius + 1, dtype=np.int32)
        acc = inner_sum[oc][None, :] + candidates[:, None]
        out = np.rint(acc * requant + CONV2_OUTPUT_ZP)
        out = np.clip(out, 0, 255).astype(np.int16)
        diff = out - ref[oc][None, :]

        mismatch = np.count_nonzero(diff, axis=1)
        abs_error = np.abs(diff).sum(axis=1)
        best_idx = int(np.argmin(mismatch * 100000 + abs_error))

        total_mismatch += int(mismatch[best_idx])
        total_abs_error += int(abs_error[best_idx])
        max_error = max(max_error, int(np.abs(diff[best_idx]).max()))

    return total_mismatch, total_abs_error, max_error


def iter_scales(start: float, stop: float, step: float) -> Iterable[float]:
    count = int(round((stop - start) / step)) + 1
    for idx in range(count):
        yield round(start + idx * step, 10)


def run_search(
    real_inputs: np.ndarray,
    weights: np.ndarray,
    ref: np.ndarray,
    weight_scale: float,
    zps: Iterable[int],
    scales: Iterable[float],
    bias: np.ndarray | None = None,
    bias_search_radius: int = 64,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for zp in zps:
        for scale in scales:
            centered_inputs = quantize_to_concat_domain(real_inputs, scale, zp)
            inner_sum = conv2_inner_sum(weights, centered_inputs)
            if bias is None:
                mismatch, abs_error, max_error = score_with_refit_bias(
                    inner_sum, ref, scale, weight_scale, bias_search_radius
                )
            else:
                mismatch, abs_error, max_error = score_with_fixed_bias(
                    inner_sum, bias, ref, scale, weight_scale
                )
            results.append(SearchResult(mismatch, abs_error, max_error, scale, zp))
    return sorted(results)


def summarize(name: str, results: list[SearchResult], top_k: int) -> None:
    best = results[0]
    same_best = [item for item in results if item.mismatch == best.mismatch and item.abs_error == best.abs_error]
    zp_candidates = sorted({item.zp for item in same_best})

    print(f"[{name}]")
    print(
        f"  best mismatch={best.mismatch} abs_error={best.abs_error} "
        f"max_error={best.max_error} scale={best.scale:.10f} zp={best.zp}"
    )
    print(f"  equally-best zp candidates: {zp_candidates}")
    print("  top results:")
    for item in results[:top_k]:
        print(
            f"    mismatch={item.mismatch:5d} abs_error={item.abs_error:5d} "
            f"max_error={item.max_error:3d} scale={item.scale:.10f} zp={item.zp}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="用 ref 数据评估 conv2 输入量化域候选参数。")
    parser.add_argument("--ref-root", default="src/ref", help="ref 根目录，默认 src/ref")
    parser.add_argument("--scale-start", type=float, default=0.0330)
    parser.add_argument("--scale-stop", type=float, default=0.0380)
    parser.add_argument("--scale-step", type=float, default=0.0001)
    parser.add_argument("--zp-min", type=int, default=0)
    parser.add_argument("--zp-max", type=int, default=15)
    parser.add_argument("--bias-search-radius", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    ref_root = Path(args.ref_root)
    bundle = load_ref_bundle(ref_root)

    scales = list(iter_scales(args.scale_start, args.scale_stop, args.scale_step))
    zps = list(range(args.zp_min, args.zp_max + 1))

    print("exported conv2_meta:")
    print(f"  input_scale={bundle['exported_scale']}")
    print(f"  input_zp={bundle['exported_zp']}")
    print(f"  search_scales=[{scales[0]:.10f}, {scales[-1]:.10f}] step={args.scale_step}")
    print(f"  search_zps=[{zps[0]}, {zps[-1]}]")
    print()

    fixed_bias_results = run_search(
        real_inputs=bundle["real_inputs"],
        weights=bundle["weights"],
        ref=bundle["conv2_ref"],
        weight_scale=bundle["weight_scale"],
        zps=zps,
        scales=scales,
        bias=bundle["bias"],
    )
    summarize("fixed exported bias", fixed_bias_results, args.top_k)

    refit_bias_results = run_search(
        real_inputs=bundle["real_inputs"],
        weights=bundle["weights"],
        ref=bundle["conv2_ref"],
        weight_scale=bundle["weight_scale"],
        zps=zps,
        scales=scales,
        bias=None,
        bias_search_radius=args.bias_search_radius,
    )
    summarize("refit per-channel bias", refit_bias_results, args.top_k)

    print("conclusion:")
    print("  1. 只靠 ref 里的 trace/weight/output，zp 通常不能唯一辨识，会被 bias 吸收。")
    print("  2. 如果想拿到真实 concat 统一量化域，还是应该直接从 PyTorch 量化后的 QFunctional 读取 scale/zero_point。")


if __name__ == "__main__":
    main()
