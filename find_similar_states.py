"""
查找训练集中与指定seed初始状态最相似的 episodes。

Usage:
  uv run --frozen python find_similar_states.py \
    --seed 100013 \
    --dataset data/pusht/pusht_cchi_v7_replay.zarr \
    --top_k 10
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import click
import numpy as np
import zarr


def circular_distance(a, b, period=2*np.pi):
    """角度环形距离，先将角度归一到 [0, period)。"""
    a = a % period
    b = b % period
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def compute_state_distance(state_a, state_b, weights=None):
    """计算两个完整5D状态之间的加权距离。

    state: [agent_x, agent_y, block_x, block_y, block_angle]
    前4维使用L2距离，第5维(角度)使用环形距离。
    """
    if weights is None:
        # 默认权重：agent_pos和block_pos等权，角度适当缩放
        weights = np.array([1.0, 1.0, 1.0, 1.0, 50.0])

    a, b = np.array(state_a), np.array(state_b)
    # 前4维: L2
    pos_dist = np.sqrt(np.sum((a[:4] - b[:4])**2))
    # 第5维: 环形距离
    ang_dist = circular_distance(a[4], b[4])
    # 加权组合
    combined = np.sqrt(
        weights[0] * (a[0]-b[0])**2 +
        weights[1] * (a[1]-b[1])**2 +
        weights[2] * (a[2]-b[2])**2 +
        weights[3] * (a[3]-b[3])**2 +
        weights[4] * ang_dist**2
    )
    return combined, pos_dist, ang_dist


def image_mse(img_a, img_b):
    """计算两张图像之间的 MSE."""
    return np.mean((img_a.astype(np.float32) - img_b.astype(np.float32))**2)


@click.command()
@click.option('-s', '--seed', required=True, type=int,
              help='目标 seed (如 100013)')
@click.option('--dataset', default='data/pusht/pusht_cchi_v7_replay.zarr',
              help='训练集 zarr 路径')
@click.option('-k', '--top_k', default=10, type=int,
              help='输出最相似的前 K 个 episodes')
@click.option('--weights', default='1,1,1,1,50',
              help='5D state 各维度权重 (agent_x,agent_y,block_x,block_y,block_angle)')
def main(seed, dataset, top_k, weights):
    # 解析权重
    w = np.array([float(x) for x in weights.split(',')], dtype=np.float64)

    # ========= 1. 生成目标 seed 的初始状态 =========
    rs = np.random.RandomState(seed=seed)
    target_state = np.array([
        rs.randint(50, 450),    # agent_x
        rs.randint(50, 450),    # agent_y
        rs.randint(100, 400),   # block_x
        rs.randint(100, 400),   # block_y
        rs.randn() * 2 * np.pi - np.pi  # block_angle
    ])
    print(f"Seed {seed} 初始状态:")
    print(f"  Agent 位置:  ({target_state[0]:.1f}, {target_state[1]:.1f})")
    print(f"  Block 位置:  ({target_state[2]:.1f}, {target_state[3]:.1f})")
    print(f"  Block 角度:  {target_state[4]:.4f} rad ({np.rad2deg(target_state[4]):.1f}°)")

    # ========= 2. 加载训练集 =========
    print(f"\n加载训练集: {dataset}")
    root = zarr.open(dataset, 'r')
    states = root['data']['state'][:]       # (N, 5)
    images = root['data']['img'][:]          # (N, 96, 96, 3)
    episode_ends = root['meta']['episode_ends'][:]  # 累计终点索引

    n_episodes = len(episode_ends)
    print(f"训练集共 {n_episodes} 个 episodes")

    # 每个 episode 的起始索引
    episode_starts = np.zeros(n_episodes, dtype=np.int64)
    episode_starts[0] = 0
    episode_starts[1:] = episode_ends[:-1]

    # ========= 3. 计算每个训练 episode 初始状态与目标状态的距离 =========
    all_dists = []
    for ep_idx in range(n_episodes):
        start_idx = episode_starts[ep_idx]
        train_state = states[start_idx]

        combined_dist, pos_dist, ang_dist = compute_state_distance(
            target_state, train_state, weights=w)

        all_dists.append({
            'episode': ep_idx,
            'start_idx': int(start_idx),
            'ep_len': int(episode_ends[ep_idx] - start_idx + 1),
            'train_state': train_state,
            'combined_dist': combined_dist,
            'pos_dist': pos_dist,
            'ang_dist': np.rad2deg(ang_dist),
        })

    # ========= 4. 排序并输出 Top-K =========
    all_dists.sort(key=lambda x: x['combined_dist'])

    print(f"\n{'='*80}")
    print(f"最相似的 Top-{top_k} 训练 episodes (seed={seed}):")
    print(f"{'='*80}")
    print(f"{'排名':<5} {'Ep':<6} {'起始Idx':<8} {'长度':<6} "
          f"{'综合距离':<12} {'位置距离':<12} {'角度差(°)':<12} "
          f"{'Agent(x,y)':<20} {'Block(x,y)':<20} {'Block角(°)'}")
    print("-" * 80)

    for rank, item in enumerate(all_dists[:top_k]):
        ts = item['train_state']
        agent_str = f"({ts[0]:.0f}, {ts[1]:.0f})"
        block_str = f"({ts[2]:.0f}, {ts[3]:.0f})"
        ang_deg = np.rad2deg(ts[4])
        print(f"{rank+1:<5} {item['episode']:<6} {item['start_idx']:<8} {item['ep_len']:<6} "
              f"{item['combined_dist']:<12.2f} {item['pos_dist']:<12.2f} {item['ang_dist']:<12.1f} "
              f"{agent_str:<20} {block_str:<20} {ang_deg:.1f}")

    # ========= 5. 额外：图像相似度 (Top-K 中的) =========
    print(f"\n{'='*80}")
    print(f"Top-{min(5, top_k)} 候选 vs 目标状态的详细对比 (含像素距离):")
    print(f"{'='*80}")

    # 渲染目标 seed 的初始图像 (需要 pygame)
    try:
        import pygame
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

        env = PushTImageEnv(render_size=96)
        env.seed(seed)
        obs = env.reset()
        target_img = obs['image']  # (3, 96, 96) float32 [0,1]
        target_img_uint8 = (target_img.transpose(1, 2, 0) * 255).astype(np.uint8)
        env.close()
        pygame.quit()

        for rank, item in enumerate(all_dists[:min(5, top_k)]):
            train_img = images[item['start_idx']]  # (96, 96, 3) uint8 [0,255]
            train_img_float = train_img.astype(np.float32) / 255.0
            # MSE in [0,1] space
            mse = image_mse(target_img.transpose(1, 2, 0), train_img_float)

            # agent pos L2
            agent_dist = np.sqrt(
                (target_state[0] - item['train_state'][0])**2 +
                (target_state[1] - item['train_state'][1])**2)
            # block pos L2
            block_dist = np.sqrt(
                (target_state[2] - item['train_state'][2])**2 +
                (target_state[3] - item['train_state'][3])**2)
            # angle difference
            ang_diff = np.rad2deg(circular_distance(target_state[4], item['train_state'][4]))

            print(f"\n  Ep {item['episode']} (idx={item['start_idx']}):")
            print(f"    Agent 距离:     {agent_dist:.1f} px")
            print(f"    Block 距离:     {block_dist:.1f} px")
            print(f"    角度差:         {ang_diff:.1f}°")
            print(f"    图像 MSE:       {mse:.6f}")
            print(f"    训练状态:       agent=({item['train_state'][0]:.0f},{item['train_state'][1]:.0f}) "
                  f"block=({item['train_state'][2]:.0f},{item['train_state'][3]:.0f}) "
                  f"angle={np.rad2deg(item['train_state'][4]):.1f}°")
            print(f"    目标状态:       agent=({target_state[0]:.0f},{target_state[1]:.0f}) "
                  f"block=({target_state[2]:.0f},{target_state[3]:.0f}) "
                  f"angle={np.rad2deg(target_state[4]):.1f}°")
    except ImportError:
        print("  (跳过图像对比: pygame 环境不可用)")


if __name__ == '__main__':
    main()
