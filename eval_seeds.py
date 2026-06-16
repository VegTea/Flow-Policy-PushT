"""
使用指定 checkpoint 在自定义 seed 列表上评估，每个 seed 保存视频。

Usage:
  python eval_seeds.py \
    --checkpoint data/outputs/.../checkpoints/latest.ckpt \
    --output_dir data/eval_output \
    --seeds 100000,100001,100005,100010 \
    --device cuda:0

生成的目录结构:
  {output_dir}/
    media/
      seed_100000.mp4
      seed_100001.mp4
      ...
    eval_log.json        # 包含每个 seed 的 reward 和视频路径
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import json
import shutil
import random
import click
import hydra
import torch
import dill
import wandb
import numpy as np
import copy

from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def parse_seeds(seeds_str: str):
    """解析 seed 列表字符串，支持逗号分隔和范围语法。

    示例:
      "100,200,300" -> [100, 200, 300]
      "0-4" -> [0, 1, 2, 3, 4]
      "0-2,10,20-22" -> [0, 1, 2, 10, 20, 21, 22]
    """
    seeds = []
    for part in seeds_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            seeds.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            seeds.append(int(part))
    return seeds


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@click.command()
@click.option('-c', '--checkpoint', required=True,
              help='Checkpoint 文件路径 (.ckpt)')
@click.option('-o', '--output_dir', required=True,
              help='输出目录，视频和日志将保存到此')
@click.option('-s', '--seeds', required=True, type=str,
              help='要评估的 seed 列表，逗号分隔，支持范围如 0-4。'
                   '示例: "100,200,300" 或 "0-4,10,20-22"')
@click.option('-d', '--device', default='cuda:0',
              help='运行设备 (默认: cuda:0)')
@click.option('--max_steps', type=int, default=None,
              help='每个 episode 的最大步数，覆盖 checkpoint 中的配置')
@click.option('--fps', type=int, default=None,
              help='视频帧率，覆盖 checkpoint 中的配置')
@click.option('--num_inference_steps', type=int, default=None,
              help='推理采样/积分步数，覆盖 checkpoint 中 policy 的配置')
@click.option('--policy_seed', type=int, default=None,
              help='固定 policy 推理随机种子，使 Flow Matching 的 torch.randn 可复现')
@click.option('--same_policy_seed/--different_policy_seed_per_env',
              default=True,
              help='多个 env seed 评估时是否复用同一个 policy_seed。默认复用；'
                   '关闭后每个 env seed 使用 policy_seed + env seed。')
def main(checkpoint, output_dir, seeds, device, max_steps, fps,
         num_inference_steps, policy_seed, same_policy_seed):
    # 解析 seeds
    seed_list = parse_seeds(seeds)
    print(f"将评估 {len(seed_list)} 个 seeds: {seed_list}")
    if policy_seed is not None:
        print(f"固定 policy_seed: {policy_seed}")
        print(f"same_policy_seed: {same_policy_seed}")

    # 创建输出目录
    if os.path.exists(output_dir):
        click.confirm(
            f"输出目录 {output_dir} 已存在！是否覆盖？", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========= 加载 checkpoint =========
    print(f"加载 checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 获取 policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    if num_inference_steps is not None:
        if not hasattr(policy, 'num_inference_steps'):
            raise AttributeError(
                f"{type(policy).__name__} does not expose num_inference_steps"
            )
        print(
            f"覆盖 num_inference_steps: "
            f"{policy.num_inference_steps} -> {num_inference_steps}"
        )
        policy.num_inference_steps = num_inference_steps

    # ========= 提取原始 env_runner 配置 =========
    runner_cfg = copy.deepcopy(cfg.task.env_runner)

    # 构建覆盖参数
    overrides = dict(
        n_train=0,
        n_train_vis=0,
        n_test=1,
        n_test_vis=1,
    )
    if max_steps is not None:
        overrides['max_steps'] = max_steps
    if fps is not None:
        overrides['fps'] = fps

    # ========= 对每个 seed 执行评估 =========
    all_video_paths = []
    all_max_rewards = []
    log_data = dict()

    for i, seed in enumerate(seed_list):
        print(f"\n[{i+1}/{len(seed_list)}] 评估 seed={seed} ...")
        if policy_seed is not None:
            rollout_policy_seed = (
                policy_seed if same_policy_seed else policy_seed + seed
            )
            set_global_seed(rollout_policy_seed)
            print(f"  rollout policy seed: {rollout_policy_seed}")

        # 为当前 seed 创建 seed-specific 输出目录 (用于视频)
        seed_output_dir = os.path.join(output_dir, f'seed_{seed}')
        pathlib.Path(seed_output_dir).mkdir(parents=True, exist_ok=True)

        # 创建 env_runner，每个 seed 只创建一个 env
        env_runner = hydra.utils.instantiate(
            runner_cfg,
            output_dir=seed_output_dir,
            test_start_seed=seed,
            **overrides,
        )

        # 运行 rollout
        runner_log = env_runner.run(policy)

        # 收集结果：先提取 reward 和视频路径
        seed_reward = None
        seed_video_src = None
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                seed_video_src = value._path
            elif key.startswith('test/sim_max_reward'):
                seed_reward = value

        # 保存视频 (文件名带上 max_reward)
        if seed_reward is not None:
            all_max_rewards.append(seed_reward)
            log_data[f"seed_{seed}_max_reward"] = seed_reward
            reward_str = f"{seed_reward:.4f}"
        else:
            reward_str = "N/A"

        if seed_video_src and os.path.exists(seed_video_src):
            dst_dir = os.path.join(output_dir, 'media')
            pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
            dst_name = f"seed_{seed}_max_reward={reward_str}.mp4"
            dst_path = os.path.join(dst_dir, dst_name)
            counter = 1
            while os.path.exists(dst_path):
                dst_name = f"seed_{seed}_max_reward={reward_str}_{counter}.mp4"
                dst_path = os.path.join(dst_dir, dst_name)
                counter += 1
            shutil.copy2(seed_video_src, dst_path)
            all_video_paths.append(dst_path)
            log_data[f"seed_{seed}_video"] = dst_path
            print(f"  max_reward: {reward_str}")
            print(f"  视频已保存: {dst_path}")
        else:
            log_data[f"seed_{seed}_video"] = None
            print(f"  max_reward: {reward_str}")

    # ========= 汇总统计 =========
    if all_max_rewards:
        log_data['mean_score'] = np.mean(all_max_rewards)
        log_data['std_score'] = np.std(all_max_rewards)
        log_data['min_score'] = np.min(all_max_rewards)
        log_data['max_score'] = np.max(all_max_rewards)
        log_data['num_seeds'] = len(seed_list)
        log_data['seeds'] = seed_list
        log_data['checkpoint'] = checkpoint
        log_data['policy_seed'] = policy_seed
        log_data['same_policy_seed'] = same_policy_seed

        print(f"\n{'='*50}")
        print(f"评估完成!")
        print(f"  Seeds 数量: {len(seed_list)}")
        print(f"  Mean Score: {log_data['mean_score']:.4f}")
        print(f"  Std Score:  {log_data['std_score']:.4f}")
        print(f"  Min Score:  {log_data['min_score']:.4f}")
        print(f"  Max Score:  {log_data['max_score']:.4f}")
        print(f"{'='*50}")

    # 保存日志
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(log_data, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"\n评估日志已保存: {out_path}")


if __name__ == '__main__':
    main()
