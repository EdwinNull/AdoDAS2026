#!/usr/bin/env python3
"""
Stage 1 消融实验链 — 按顺序跑 4 组实验，每组等待 GPU 空闲 ≥27GB 后启动

实验顺序:
  1. default                  (基线)
  2. default + lupi both      (基线 + LUPI 全开)
  3. full                     (全量 MTL, 不做训后推理)
  4. full + lupi both         (全量 MTL + LUPI 全开)

用法:
  python scripts/run_ablation.py                          # 全部 4 组
  python scripts/run_ablation.py --start 2                # 从第 2 组开始
  python scripts/run_ablation.py --only 1                 # 只跑第 1 组
  python scripts/run_ablation.py --free-gb 20 --gpu 0     # 自定义 GPU 参数
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Experiment:
    name: str
    preset: str
    lupi: str          # "" | "heads" | "weight" | "both"
    extra: list[str] = field(default_factory=list)

    def build_cmd(self) -> list[str]:
        cmd = [
            "python", "train.py",
            "--task", "a2",
            "--config", f"tasks/a2/{'mtl_full' if self.preset == 'full' else 'default'}.yaml",
            "--feature_root", "/data1/AdoDas",
            "--manifest_dir", "/data1/AdoDas",
            "--output_dir", "./output",
            "--use_hdf5", "1",
            "--preload", "1",
        ]
        if self.lupi:
            cmd += ["--aux_lupi_enabled", "1"]
            if self.lupi in ("heads", "both"):
                cmd += ["--aux_lupi_heads", "1"]
            if self.lupi in ("weight", "both"):
                cmd += ["--aux_lupi_reweight", "1"]
        if self.extra:
            cmd += self.extra
        return cmd


EXPERIMENTS: list[Experiment] = [
    Experiment(name="default",           preset="default", lupi=""),
    Experiment(name="default+lupi_both", preset="default", lupi="both"),
    Experiment(name="full",              preset="full",    lupi=""),
    Experiment(name="full+lupi_both",    preset="full",    lupi="both"),
]


def query_free_memory(gpu_ids: list[int] | None = None) -> dict[int, int]:
    """查询 GPU 空闲显存 (MB)，返回 {gpu_index: free_mb}。gpu_ids=None 时返回全部 GPU。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    free = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        idx, mb = line.split(",")
        i = int(idx.strip())
        if gpu_ids is None or i in gpu_ids:
            free[i] = int(mb.strip())
    return free


def wait_for_gpu(gpu_ids: list[int] | None, free_gb: float, idle_duration: int,
                 check_interval: int = 5) -> int:
    """等待任一张 GPU 空闲达标并持续稳定，返回选中的 GPU ID"""
    target_mb = int(free_gb * 1024)
    idle_start: float | None = None
    label = f"GPU {','.join(map(str, gpu_ids))}" if gpu_ids else "全部 GPU"

    print(f"  等待 {label} 空闲 ≥ {free_gb:.0f}GB, 持续 {idle_duration}s ...")
    while True:
        free = query_free_memory(gpu_ids)
        if not free:
            print("  nvidia-smi 查询失败, 重试...", file=sys.stderr)
            time.sleep(check_interval)
            continue

        target_free = max(free.values())
        best_gpu = max(free, key=free.get)
        ready = target_free >= target_mb
        t = time.strftime("%H:%M:%S")
        gb_str = ", ".join(f"GPU{g}: {m/1024:.1f}GB" for g, m in sorted(free.items()))
        print(f"  [{t}] {gb_str}  {'✓ 达标' if ready else '  等待'}")

        if ready:
            if idle_start is None:
                idle_start = time.time()
            elif time.time() - idle_start >= idle_duration:
                print(f"  -> 选中 GPU {best_gpu} ({free[best_gpu]/1024:.1f}GB 空闲), 启动训练")
                return best_gpu
        else:
            idle_start = None
        time.sleep(check_interval)


def run_one(exp: Experiment, idx: int, total: int, args, gpu_ids: list[int] | None) -> int:
    """运行单个实验, 返回退出码"""
    print(f"\n{'=' * 70}")
    print(f"  实验 {idx}/{total}: {exp.name}")
    print(f"{'=' * 70}")

    cmd = exp.build_cmd()
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        print("  [dry-run] 跳过等待和执行\n")
        return 0

    selected_gpu = wait_for_gpu(
        gpu_ids=gpu_ids,
        free_gb=args.free_gb,
        idle_duration=args.idle_duration,
    )

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(selected_gpu)}

    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, env=env)
        ret = proc.wait()
    except KeyboardInterrupt:
        print("\n用户中断, 终止当前实验...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    elapsed = time.time() - t0
    status = "✓ 成功" if ret == 0 else f"✗ 失败 (exit={ret})"
    print(f"\n  实验 {idx}/{total} ({exp.name}): {status}  耗时 {elapsed/60:.1f}min")
    return ret


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 消融实验链：default → +lupi → full → full+lupi")
    parser.add_argument("--start", type=int, default=1,
                        help="从第几组开始 (1-based, 默认 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="只跑第 N 组 (1-based)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU ID 列表 (逗号分隔, 默认: 监控全部 GPU)")
    parser.add_argument("--free-gb", type=float, default=28.0,
                        help="需要的空闲显存 GB (默认 28=27GB训练+1GB余量)")
    parser.add_argument("--idle-duration", type=int, default=30,
                        help="空闲持续秒数 (默认 30)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="任一组失败则中止后续实验")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印命令，不实际等待 GPU 或执行训练")
    args = parser.parse_args()

    gpu_ids = None
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]

    exps = EXPERIMENTS
    if args.only is not None:
        if not (1 <= args.only <= len(exps)):
            print(f"错误: --only 范围 1-{len(exps)}")
            return 1
        exps = [exps[args.only - 1]]
        start_label = args.only
    else:
        exps = exps[args.start - 1:]
        start_label = args.start

    total = len(exps)
    gpu_label = ",".join(map(str, gpu_ids)) if gpu_ids else "全部"
    print("=" * 70)
    print(f"  Stage 1 消融实验链  共 {total} 组  "
          f"GPU={gpu_label}  空闲阈值={args.free_gb:.0f}GB  冷却={args.idle_duration}s")
    print("=" * 70)
    for i, exp in enumerate(exps):
        print(f"  {start_label + i}. {exp.name}")
    print("=" * 70)

    results: list[tuple[str, int]] = []
    for i, exp in enumerate(exps):
        try:
            ret = run_one(exp, start_label + i, start_label + total - 1, args, gpu_ids)
            results.append((exp.name, ret))
            if ret != 0 and args.stop_on_error:
                print(f"\n  中止: {exp.name} 失败且 --stop-on-error 已启用")
                break
        except KeyboardInterrupt:
            print("\n  用户中断, 实验链终止")
            results.append((exp.name, -1))
            break

    print("\n" + "=" * 70)
    print("  消融实验链完成")
    print("=" * 70)
    for name, ret in results:
        s = "✓" if ret == 0 else "✗"
        print(f"  {s}  {name}")
    print("=" * 70)

    all_ok = all(r == 0 for _, r in results)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
