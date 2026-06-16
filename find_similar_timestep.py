"""
在指定 seed 的策略 rollout 中，捕获某时刻的状态，并搜索训练集中最相似的状态。

Usage:
  uv run --frozen python find_similar_timestep.py \
    --checkpoint data/outputs/.../checkpoints/epoch=0400-test_mean_score=0.869.ckpt \
    --seed 100013 \
    --time 9.0 \
    --top_k 10
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import click
import numpy as np
import zarr
import torch
import dill
import hydra
import copy
from collections import OrderedDict


def circular_distance(a, b, period=2 * np.pi):
    """角度环形距离，先将角度归一到 [0, period)。"""
    a = a % period
    b = b % period
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def image_mse(img_a, img_b):
    """两张图像之间的 MSE (值域 [0,1])."""
    return float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))


@click.command()
@click.option('-c', '--checkpoint', required=True,
              help='Checkpoint 文件路径 (.ckpt)')
@click.option('-s', '--seed', required=True, type=int,
              help='评估 seed')
@click.option('-t', '--time', 'time_sec', required=True, type=float,
              help='目标时刻 (秒)，fps=10 所以 t=9.0 对应第 90 步')
@click.option('--dataset', default='data/pusht/pusht_cchi_v7_replay.zarr',
              help='训练集 zarr 路径')
@click.option('-k', '--top_k', default=10, type=int,
              help='输出最相似的前 K 个 timesteps')
@click.option('-d', '--device', default='cuda:0',
              help='运行设备')
def main(checkpoint, seed, time_sec, dataset, top_k, device):
    fps = 10
    target_step = int(time_sec * fps)

    # ============================================================
    # 1. 加载 checkpoint 和 policy
    # ============================================================
    print(f"加载 checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps

    # ============================================================
    # 2. 运行 rollout 到目标时刻，捕获状态
    # ============================================================
    print(f"运行 rollout: seed={seed}, 目标 t={time_sec}s (step={target_step})")

    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = PushTImageEnv(render_size=96)
    env.seed(seed)
    env.reset()  # 这会设置初始状态
    # 获取初始完整状态 (PushTImageEnv._get_obs 返回 dict, 需要从底层 body 获取)
    init_full_state = np.array([
        env.agent.position[0], env.agent.position[1],
        env.block.position[0], env.block.position[1],
        env.block.angle
    ])
    print(f"初始状态: agent=({init_full_state[0]:.1f},{init_full_state[1]:.1f}) "
          f"block=({init_full_state[2]:.1f},{init_full_state[3]:.1f}) "
          f"angle={np.rad2deg(init_full_state[4]):.1f}°")

    # 重新创建 env，使用 MultiStepWrapper 来获得正确的 obs 格式
    env.close()
    import pygame
    pygame.quit()

    def make_env():
        env = PushTImageEnv(render_size=96, legacy=True)
        env.seed(seed)
        return env

    wrapped_env = MultiStepWrapper(
        make_env(),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=10000,
    )
    wrapped_env.seed(seed)

    obs = wrapped_env.reset()

    past_action = None
    policy.reset()

    captured_image = None        # (H, W, 3) uint8  [0,255]
    captured_agent_pos = None    # (2,) float32
    captured_full_state = None   # (5,) float64  [agent_x, agent_y, block_x, block_y, block_angle]

    steps_done = 0
    done = False
    while not done:
        # 到达目标步数时捕获状态
        if steps_done == target_step:
            # 从底层 env 获取完整 5D 状态
            # PushTImageEnv 覆写了 _get_obs 返回 dict，需直接从 body 读取
            base_env = wrapped_env.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            # base_env 现在是 PushTImageEnv (继承 PushTEnv)
            captured_full_state = np.array([
                base_env.agent.position[0], base_env.agent.position[1],
                base_env.block.position[0], base_env.block.position[1],
                base_env.block.angle
            ])

            # 从 obs 获取 image 和 agent_pos
            # obs['image'] 是 (n_obs_steps, 3, 96, 96) 或类似格式
            # 取最后一步
            if 'image' in obs:
                img_data = obs['image']
                if img_data.ndim == 4:  # (n_obs_steps, 3, H, W)
                    img_data = img_data[-1]  # (3, H, W)
                captured_image = (img_data.transpose(1, 2, 0) * 255).astype(np.uint8)
            if 'agent_pos' in obs:
                ap = obs['agent_pos']
                if ap.ndim == 2:  # (n_obs_steps, 2)
                    ap = ap[-1]
                captured_agent_pos = ap.copy()

            print(f"\n捕获 t={time_sec}s (step={target_step}) 的状态:")
            print(f"  Agent 位置: ({captured_full_state[0]:.1f}, {captured_full_state[1]:.1f})")
            print(f"  Block 位置: ({captured_full_state[2]:.1f}, {captured_full_state[3]:.1f})")
            print(f"  Block 角度: {captured_full_state[4]:.4f} rad ({np.rad2deg(captured_full_state[4]):.1f}°)")

        # 构建 obs_dict (匹配 policy 期望的 batch 维度)
        np_obs_dict = dict(obs)
        obs_dict = dict()
        for k, v in np_obs_dict.items():
            t = torch.from_numpy(v).to(device=device, dtype=torch.float32)
            # MultiStepWrapper 返回 (n_obs_steps, ...)，policy 期望 (B, n_obs_steps, ...)
            t = t.unsqueeze(0)  # 添加 batch dim
            obs_dict[k] = t

        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        action = action_dict['action'].detach().cpu().numpy()
        if action.ndim == 3:
            action = action[0]  # (n_action_steps, 2)

        obs, reward, done, info = wrapped_env.step(action)
        done = np.all(done) if isinstance(done, np.ndarray) else done
        past_action = action
        steps_done += 1

    wrapped_env.close()
    import pygame
    pygame.quit()
    print(f"\nRollout 完成，共 {steps_done} 步，目标步已捕获: {captured_full_state is not None}")

    if captured_full_state is None:
        print(f"错误: rollout 在 step {target_step} 之前就结束了! (总共仅 {steps_done} 步)")
        return

    # ============================================================
    # 3. 加载训练集，搜索所有 timesteps
    # ============================================================
    print(f"\n加载训练集: {dataset}")
    root = zarr.open(dataset, 'r')
    all_states = root['data']['state'][:]       # (25650, 5)
    all_images = root['data']['img'][:]          # (25650, 96, 96, 3)
    episode_ends = root['meta']['episode_ends'][:]

    n_total = len(all_states)

    # 只搜索训练集中实际使用的 timesteps (90 个最大 episodes)
    from diffusion_policy.common.sampler import get_val_mask, downsample_mask
    n_episodes = len(episode_ends)
    val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=0.02, seed=42)
    train_mask = downsample_mask(mask=~val_mask, max_n=90, seed=42)
    train_ep_indices = np.where(train_mask)[0]

    # 构建训练集 step 索引范围
    ep_starts = np.zeros(n_episodes, dtype=np.int64)
    ep_starts[0] = 0
    ep_starts[1:] = episode_ends[:-1]

    train_step_mask = np.zeros(n_total, dtype=bool)
    for ep_idx in train_ep_indices:
        s = ep_starts[ep_idx]
        e = episode_ends[ep_idx]
        train_step_mask[s:e + 1] = True

    n_train_steps = train_step_mask.sum()
    print(f"搜索范围: {n_train_steps} 个训练 timesteps (来自 {len(train_ep_indices)} 个训练 episodes)")

    # ============================================================
    # 4. 计算距离
    # ============================================================
    target_state = captured_full_state.astype(np.float64)
    target_img = captured_image.astype(np.float32) / 255.0  # (96,96,3) [0,1]

    # 向量化计算 5D state 距离
    train_states = all_states[train_step_mask].astype(np.float64)
    train_images = all_images[train_step_mask]
    train_indices = np.where(train_step_mask)[0]

    # Agent+Block L2 (前4维)
    pos_diff = train_states[:, :4] - target_state[:4]
    pos_dist = np.sqrt(np.sum(pos_diff ** 2, axis=1))

    # 角度环形距离
    target_angle = target_state[4] % (2 * np.pi)
    train_angles = train_states[:, 4] % (2 * np.pi)
    ang_diff = np.abs(train_angles - target_angle)
    ang_dist_rad = np.minimum(ang_diff, 2 * np.pi - ang_diff)

    # 加权综合距离 (权重: agent 1,1; block 1,1; angle 50)
    w_angle = 50.0
    combined_dist = np.sqrt(pos_diff[:, 0] ** 2 + pos_diff[:, 1] ** 2 +
                            pos_diff[:, 2] ** 2 + pos_diff[:, 3] ** 2 +
                            (w_angle * ang_dist_rad) ** 2)

    # 排序
    sorted_idx = np.argsort(combined_dist)

    # ============================================================
    # 5. 输出 Top-K
    # ============================================================
    print(f"\n{'=' * 90}")
    print(f"  最相似的 Top-{top_k} 训练 timesteps (seed={seed}, t={time_sec}s)")
    print(f"{'=' * 90}")
    print(f"{'排名':<5} {'全局Idx':<8} {'Ep':<6} {'Ep内步':<8} "
          f"{'综合距离':<10} {'位置距离':<10} {'角度差°':<8} "
          f"{'Agent(x,y)':<20} {'Block(x,y)':<20} {'Block角°'}")
    print("-" * 90)

    for rank in range(top_k):
        k = sorted_idx[rank]
        global_idx = train_indices[k]
        dist = combined_dist[k]
        p_dist = pos_dist[k]
        a_dist = np.rad2deg(ang_dist_rad[k])
        ts = train_states[k]

        # 找到该索引属于哪个 episode 和 episode 内步数
        ep_idx = np.searchsorted(episode_ends, global_idx, side='right')
        ep_start = ep_starts[ep_idx]
        step_in_ep = global_idx - ep_start

        agent_str = f"({ts[0]:.0f}, {ts[1]:.0f})"
        block_str = f"({ts[2]:.0f}, {ts[3]:.0f})"
        ang_deg = np.rad2deg(ts[4] % (2 * np.pi))

        print(f"{rank + 1:<5} {global_idx:<8} {ep_idx:<6} {step_in_ep:<8} "
              f"{dist:<10.2f} {p_dist:<10.2f} {a_dist:<8.1f} "
              f"{agent_str:<20} {block_str:<20} {ang_deg:.1f}")

    # ============================================================
    # 6. 详细对比 Top-5 (含图像 MSE)
    # ============================================================
    print(f"\n{'=' * 90}")
    print(f"  Top-{min(5, top_k)} 详细对比 (含图像 MSE):")
    print(f"{'=' * 90}")

    for rank in range(min(5, top_k)):
        k = sorted_idx[rank]
        global_idx = train_indices[k]
        ts = train_states[k]
        ti = train_images[k]  # (96, 96, 3) uint8
        ti_float = ti.astype(np.float32) / 255.0

        mse = image_mse(target_img, ti_float)

        ep_idx = np.searchsorted(episode_ends, global_idx, side='right')
        ep_start = ep_starts[ep_idx]
        step_in_ep = global_idx - ep_start

        agent_dist = np.sqrt((ts[0] - target_state[0]) ** 2 + (ts[1] - target_state[1]) ** 2)
        block_dist = np.sqrt((ts[2] - target_state[2]) ** 2 + (ts[3] - target_state[3]) ** 2)
        ang_d = np.rad2deg(circular_distance(ts[4], target_state[4]))

        print(f"\n  [{rank + 1}] Ep {ep_idx} | 全局 idx={global_idx} | Episode 内 step={step_in_ep}")
        print(f"      训练状态: agent=({ts[0]:.0f},{ts[1]:.0f}) block=({ts[2]:.0f},{ts[3]:.0f}) angle={np.rad2deg(ts[4] % (2 * np.pi)):.1f}°")
        print(f"      目标状态: agent=({target_state[0]:.0f},{target_state[1]:.0f}) block=({target_state[2]:.0f},{target_state[3]:.0f}) angle={np.rad2deg(target_state[4] % (2 * np.pi)):.1f}°")
        print(f"      Agent 距离: {agent_dist:.1f} px | Block 距离: {block_dist:.1f} px | 角度差: {ang_d:.1f}°")
        print(f"      图像 MSE:   {mse:.6f}")


if __name__ == '__main__':
    main()
