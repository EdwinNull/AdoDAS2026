#!/usr/bin/env python3
"""
GPU 显存监控 — 空闲显存达到阈值后自动启动训练

使用示例:
  # 等待任一 GPU 空闲 ≥27GB，稳定 30 秒后启动 A2 默认训练
  python scripts/gpu_monitor_train.py --command "./run_train.sh --task a2 --preset default"

  # 指定 GPU 0，空闲 60 秒，20GB 阈值
  python scripts/gpu_monitor_train.py --gpu 0 --free-gb 20 --idle-duration 60 \\
      --command "python train.py --task a2 --config tasks/a2/mtl_full.yaml"

  # 监控多卡，取最大空闲者
  python scripts/gpu_monitor_train.py --gpu 0 1 2 3 --free-gb 27 \\
      --command "./run_train.sh --task a2 --preset full --lupi both"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class GPUInfo:
    index: int
    name: str
    util_pct: float
    mem_total_mb: int
    mem_used_mb: int
    mem_free_mb: int


def query_gpus(gpu_ids: list[int] | None = None) -> list[GPUInfo]:
    """查询 GPU 状态，返回指定 ID 或全部 GPU 的信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"nvidia-smi 查询失败: {e}", file=sys.stderr)
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        idx = int(parts[0])
        if gpu_ids is not None and idx not in gpu_ids:
            continue
        gpus.append(GPUInfo(
            index=idx,
            name=parts[1],
            util_pct=float(parts[2]),
            mem_total_mb=int(parts[3]),
            mem_used_mb=int(parts[4]),
            mem_free_mb=int(parts[5]),
        ))
    return gpus


def format_gpu_row(info: GPUInfo, target_free_mb: int) -> str:
    free_gb = info.mem_free_mb / 1024
    flag = " <-- READY" if info.mem_free_mb >= target_free_mb else ""
    return (f"  GPU {info.index} ({info.name}): "
            f"util={info.util_pct:.0f}%  "
            f"free={free_gb:.1f}GB / {info.mem_total_mb / 1024:.0f}GB{flag}")


def main():
    parser = argparse.ArgumentParser(
        description="等待 GPU 空闲显存达到阈值后自动启动训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --command "./run_train.sh --task a2 --preset default"
  %(prog)s --gpu 0 --free-gb 27 --idle-duration 60 --command "python train.py ..."
        """,
    )
    parser.add_argument("--gpu", type=int, nargs="*", default=None,
                        help="目标 GPU ID（默认：所有 GPU 中任一张满足即可）")
    parser.add_argument("--free-gb", type=float, default=28.0,
                        help="需要的空闲显存 (GB)（默认 28=27GB训练+1GB余量）")
    parser.add_argument("--idle-duration", type=int, default=30,
                        help="连续空闲多久后启动 (秒)（默认 30）")
    parser.add_argument("--check-interval", type=int, default=5,
                        help="检查间隔 (秒)（默认 5）")
    parser.add_argument("--command", type=str, required=True,
                        help="达到条件后执行的训练命令，用引号包裹")
    args = parser.parse_args()

    target_free_mb = int(args.free_gb * 1024)
    gpu_ids = args.gpu  # None = all GPUs
    idle_start: float | None = None

    print("=" * 70)
    print(f"GPU 显存监控 — 等待空闲 ≥ {args.free_gb:.0f}GB，"
          f"持续空闲 {args.idle_duration}s 后启动训练")
    print(f"训练命令: {args.command}")
    print("=" * 70)

    while True:
        gpus = query_gpus(gpu_ids)
        if not gpus:
            print("未检测到 GPU，重试中...", file=sys.stderr)
            time.sleep(args.check_interval)
            continue

        # 判断：任一张 GPU 空闲显存 ≥ 目标
        any_ready = any(g.mem_free_mb >= target_free_mb for g in gpus)

        # 显示状态
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}]")
        for g in gpus:
            print(format_gpu_row(g, target_free_mb))

        if any_ready:
            if idle_start is None:
                idle_start = time.time()
                print(f"  -> GPU 空闲显存达标，开始计时 "
                      f"({args.idle_duration}s 后启动)...")
            else:
                elapsed = time.time() - idle_start
                remaining = args.idle_duration - elapsed
                if remaining <= 0:
                    print(f"\n  GPU 已持续空闲 {elapsed:.0f}s，启动训练!")
                    break
                else:
                    print(f"  -> 已空闲 {elapsed:.0f}s，还需 {remaining:.0f}s...")
        else:
            if idle_start is not None:
                print("  -> GPU 不再满足条件，计时重置")
            idle_start = None

        time.sleep(args.check_interval)

    # 选择空闲显存最大的 GPU
    best_gpu = max(gpus, key=lambda g: g.mem_free_mb)
    env = {
        **sys.modules.get("os", __import__("os")).environ,
        "CUDA_VISIBLE_DEVICES": str(best_gpu.index),
    }

    print("=" * 70)
    print(f"使用 GPU {best_gpu.index} (空闲 {best_gpu.mem_free_mb / 1024:.1f}GB)")
    print(f"执行: {args.command}")
    print("=" * 70)

    try:
        proc = subprocess.Popen(args.command, shell=True, env=env)
        return_code = proc.wait()
        print(f"\n训练结束，返回码: {return_code}")
        return return_code
    except KeyboardInterrupt:
        print("\n用户中断")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 1


if __name__ == "__main__":
    sys.exit(main())
