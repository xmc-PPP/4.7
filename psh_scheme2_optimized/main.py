"""
主运行脚本 - 方案2: 统一DRL框架，混合动作空间

框架:
- 单一DRL智能体统一决策PSH和BESS
- 混合动作空间: PSH离散 + BESS连续
- PSH动作: {保持, 启动发电, 启动抽水, 停止}
- BESS动作: 连续[-1, 1]

储能配置:
- PSH: 34节点，定速运行，上下水库
- BESS1: 16节点，15分钟调节
- BESS2: 27节点，15分钟调节
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, '/mnt/okcomputer/output/psh_scheme2')

from utils.data_generator import (
    generate_ieee34_node_data,
    create_ieee34_node_topology,
    save_data
)
from envs.distribution_network import DistributionNetworkEnv
from algorithms.ddpg import DDPGAgent, DDPGTrainer
from configs.config import (
    psh_config, bess_configs, env_config,
    ddpg_config, training_config, eval_config
)


def set_random_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_data():
    """生成训练和测试数据"""
    print("=" * 60)
    print("步骤 1: 生成IEEE 34节点配电网数据")
    print("=" * 60)

    load_data, pv_data, price_data = generate_ieee34_node_data()
    node_data, line_data = create_ieee34_node_topology()

    data_dir = "/mnt/okcomputer/output/psh_scheme2/data"
    save_data(load_data, pv_data, price_data, node_data, line_data, data_dir)

    print(f"\n数据生成完成!")
    print(f"  - 负荷数据: {load_data.shape}")
    print(f"  - 光伏数据: {pv_data.shape}")
    print(f"  - 电价数据: {price_data.shape}")


def create_environment():
    """创建环境"""
    print("\n" + "=" * 60)
    print("步骤 2: 创建配电网环境")
    print("=" * 60)

    data_dir = "/mnt/okcomputer/output/psh_scheme2/data"
    load_data = pd.read_csv(f"{data_dir}/load_data.csv", index_col=0, parse_dates=True)
    pv_data = pd.read_csv(f"{data_dir}/pv_data.csv", index_col=0, parse_dates=True)
    price_data = pd.read_csv(f"{data_dir}/price_data.csv", index_col=0, parse_dates=True)

    psh_cfg = {
        'unit_id': psh_config.unit_id,
        'node_id': psh_config.node_id,
        'rated_generation_power': psh_config.rated_generation_power,
        'rated_pumping_power': psh_config.rated_pumping_power,
        'upper_reservoir_capacity': psh_config.upper_reservoir_capacity,
        'lower_reservoir_capacity': psh_config.lower_reservoir_capacity,
        'upper_reservoir_min': psh_config.upper_reservoir_min,
        'lower_reservoir_min': psh_config.lower_reservoir_min,
        'generation_efficiency': psh_config.generation_efficiency,
        'pumping_efficiency': psh_config.pumping_efficiency,
        'initial_upper_soc': psh_config.initial_upper_soc,
        'initial_lower_soc': psh_config.initial_lower_soc,
        'max_daily_cycles': psh_config.max_daily_cycles,
        'min_operation_duration': psh_config.min_operation_duration,
        'max_operation_duration': psh_config.max_operation_duration,
    }

    bess_cfgs = []
    for cfg in bess_configs:
        bess_cfgs.append({
            'unit_id': cfg.unit_id,
            'node_id': cfg.node_id,
            'max_power': cfg.max_power,
            'capacity': cfg.capacity,
            'min_soc': cfg.min_soc,
            'max_soc': cfg.max_soc,
            'charge_efficiency': cfg.charge_efficiency,
            'discharge_efficiency': cfg.discharge_efficiency,
            'initial_soc': cfg.initial_soc,
            'ramp_rate_limit': cfg.ramp_rate_limit
        })

    env = DistributionNetworkEnv(
        node_file=f"{data_dir}/node_data.csv",
        line_file=f"{data_dir}/line_data.csv",
        load_data=load_data,
        price_data=price_data,
        pv_data=pv_data,
        psh_config=psh_cfg,
        bess_configs=bess_cfgs,
        time_step=env_config.time_step,
        episode_length=env_config.episode_length
    )

    print("环境创建成功!")
    print(f"  - 状态维度: {env.state_dim}")
    print(f"  - 动作维度: {env.action_dim}")
    print(f"  - 时间步长: {env.time_step} 小时")
    print(f"  - 回合长度: {env.episode_length} 步")
    print(f"\n储能配置:")
    print(f"  - PSH: 节点{psh_config.node_id}, 定速运行")
    print(f"    * 额定发电功率: {psh_config.rated_generation_power} MW")
    print(f"    * 额定抽水功率: {psh_config.rated_pumping_power} MW")
    print(f"    * 上水库容量: {psh_config.upper_reservoir_capacity} MWh")
    print(f"    * 下水库容量: {psh_config.lower_reservoir_capacity} MWh")
    print(f"    * 每日启停限制: ≤{psh_config.max_daily_cycles}次")
    print(f"    * 运行时长: {psh_config.min_operation_duration*0.25}-{psh_config.max_operation_duration*0.25}小时")
    print(f"  - BESS1: 节点{bess_configs[0].node_id}, {bess_configs[0].max_power}MW/{bess_configs[0].capacity}MWh")
    print(f"  - BESS2: 节点{bess_configs[1].node_id}, {bess_configs[1].max_power}MW/{bess_configs[1].capacity}MWh")
    print(f"\n动作空间 (混合):")
    print(f"  - action[0]: PSH离散 (连续[-1,1]映射到{{0:保持, 1:启动发电, 2:启动抽水, 3:停止}})")
    print(f"  - action[1]: BESS1连续 [-1, 1]")
    print(f"  - action[2]: BESS2连续 [-1, 1]")

    return env


def train():
    """训练DDPG智能体"""
    print("\n" + "=" * 60)
    print("步骤 3: 训练DDPG智能体 (统一决策PSH和BESS)")
    print("=" * 60)

    set_random_seed(training_config.random_seed)

    env = create_environment()

    agent = DDPGAgent(
        state_dim=ddpg_config.state_dim,
        action_dim=ddpg_config.action_dim,
        actor_lr=ddpg_config.actor_lr,
        critic_lr=ddpg_config.critic_lr,
        gamma=ddpg_config.gamma,
        tau=ddpg_config.tau,
        buffer_capacity=ddpg_config.buffer_capacity,
        batch_size=ddpg_config.batch_size,
        hidden_dims=ddpg_config.hidden_dims,
        warmup_steps=ddpg_config.warmup_steps,
        device=ddpg_config.device
    )

    print(f"\nDDPG智能体配置:")
    print(f"  - Actor学习率: {ddpg_config.actor_lr}")
    print(f"  - Critic学习率: {ddpg_config.critic_lr}")
    print(f"  - 折扣因子: {ddpg_config.gamma}")
    print(f"  - 软更新系数: {ddpg_config.tau}")
    print(f"  - 回放缓冲区: {ddpg_config.buffer_capacity}")
    print(f"  - 批量大小: {ddpg_config.batch_size}")
    print(f"  - 预热步数: {ddpg_config.warmup_steps}")
    print(f"  - 隐藏层: {ddpg_config.hidden_dims}")
    print(f"  - 设备: {ddpg_config.device}")

    trainer = DDPGTrainer(
        env=env,
        agent=agent,
        max_episodes=training_config.max_episodes,
        max_steps_per_episode=training_config.max_steps_per_episode,
        eval_interval=training_config.eval_interval,
        save_interval=training_config.save_interval,
        log_interval=training_config.log_interval
    )

    print(f"\n开始训练 {training_config.max_episodes} 个回合...")
    print("注意: 单一DRL智能体统一决策PSH和BESS")
    start_time = datetime.now()

    trainer.train()

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 3600

    print(f"\n训练完成! 用时: {training_time:.2f} 小时")

    os.makedirs("models", exist_ok=True)
    agent.save(training_config.model_save_path)
    print(f"最终模型已保存到 {training_config.model_save_path}")

    print("\n绘制训练历史...")
    trainer.plot_training_history()

    results_dir = eval_config.results_dir
    os.makedirs(results_dir, exist_ok=True)

    np.save(f"{results_dir}/episode_rewards.npy", trainer.episode_rewards)
    np.save(f"{results_dir}/eval_rewards.npy", trainer.eval_rewards)
    np.save(f"{results_dir}/voltage_violations.npy", trainer.episode_voltage_violations)

    stats_df = pd.DataFrame({
        'episode': range(1, len(trainer.episode_rewards) + 1),
        'reward': trainer.episode_rewards,
        'voltage_violations': trainer.episode_voltage_violations,
        'psh_upper_soc_mean': [s['psh_upper_mean'] for s in trainer.episode_soc_stats],
        'psh_lower_soc_mean': [s['psh_lower_mean'] for s in trainer.episode_soc_stats],
        'bess1_soc_mean': [s['bess1_mean'] for s in trainer.episode_soc_stats],
        'bess2_soc_mean': [s['bess2_mean'] for s in trainer.episode_soc_stats]
    })
    stats_df.to_csv(f"{results_dir}/training_stats.csv", index=False)

    print(f"训练统计已保存到 {results_dir}")
    print(f"  - 平均电压越限次数: {np.mean(trainer.episode_voltage_violations):.2f}")
    print(f"  - 最后100回合平均奖励: {np.mean(trainer.episode_rewards[-100:]):.4f}")

    return agent, env


def evaluate(model_path: str, num_episodes: int = 10):
    """评估训练好的模型"""
    print("\n" + "=" * 60)
    print("步骤 4: 评估模型")
    print("=" * 60)

    env = create_environment()

    agent = DDPGAgent(
        state_dim=ddpg_config.state_dim,
        action_dim=ddpg_config.action_dim,
        device=ddpg_config.device
    )

    print(f"加载模型: {model_path}")
    agent.load(model_path)

    print(f"\n评估 {num_episodes} 个回合...")

    all_rewards = []
    all_violations = []
    all_psh_actions = []
    all_psh_upper_socs = []
    all_psh_lower_socs = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_violations = 0
        episode_psh_actions = []
        episode_psh_upper_socs = []
        episode_psh_lower_socs = []

        for step in range(env.episode_length):
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_violations += info.get('v_violation_count', 0)
            episode_psh_actions.append(info['psh_action'])
            episode_psh_upper_socs.append(info['psh_upper_soc'])
            episode_psh_lower_socs.append(info['psh_lower_soc'])

            state = next_state
            if done:
                break

        all_rewards.append(episode_reward)
        all_violations.append(episode_violations)
        all_psh_actions.extend(episode_psh_actions)

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.4f}, Violations = {episode_violations}")

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_violations = np.mean(all_violations)

    print(f"\n评估结果:")
    print(f"  平均奖励: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  平均电压越限次数: {mean_violations:.2f}")

    # PSH动作统计
    from collections import Counter
    action_counts = Counter(all_psh_actions)
    action_names = {0: 'HOLD', 1: 'GEN', 2: 'PUMP', 3: 'STOP'}
    print(f"\nPSH动作统计:")
    for action, count in sorted(action_counts.items()):
        print(f"  - {action_names[action]}: {count}")

    return all_rewards, all_violations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Unified DRL Framework for PSH and BESS Dispatch'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['generate_data', 'train', 'eval', 'all'],
        help='运行模式'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/ddpg_model.pth',
        help='模型路径'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=10,
        help='评估回合数'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("抽水储能与电池储能协同调度 - 方案2: 统一DRL框架")
    print("=" * 70)
    print("\n框架特点:")
    print("  • 单一DRL智能体统一决策PSH和BESS")
    print("  • 混合动作空间: PSH离散 + BESS连续")
    print("  • PSH动作: {0:保持, 1:启动发电, 2:启动抽水, 3:停止}")
    print("  • BESS动作: 连续[-1, 1]")
    print("\n储能配置:")
    print(f"  • PSH: 节点34, 定速{psh_config.rated_generation_power}MW, 上下水库")
    print(f"  • BESS1: 节点16, {bess_configs[0].max_power}MW/{bess_configs[0].capacity}MWh")
    print(f"  • BESS2: 节点27, {bess_configs[1].max_power}MW/{bess_configs[1].capacity}MWh")
    print("=" * 70)

    if args.mode == 'generate_data':
        generate_data()
    elif args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate(args.model_path, args.num_eval_episodes)
    elif args.mode == 'all':
        generate_data()
        agent, env = train()
        evaluate(training_config.model_save_path, args.num_eval_episodes)

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
