"""
主运行脚本 - 版本4.9.2 (修复版)

重大改进:
1. 修复NaN损失问题
2. 降低学习率到1e-5
3. 禁用动态奖励归一化，使用固定缩放
4. 添加更严格的NaN检查
5. 增强可视化 - 包含所有损失值
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/mnt/okcomputer/output/4.9.2')

from envs.distribution_network import DistributionNetworkEnv
from algorithms.ppo import PPOAgent, PPOTrainer


def set_random_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_environment():
    """创建环境"""
    print("\n" + "=" * 70)
    print("创建配电网环境 - 版本4.9.2 (修复版)")
    print("=" * 70)
    
    # 数据文件路径
    node_file = "Nodes_34.csv"
    line_file = "Lines_34.csv"
    time_series_file = "34_node_time_series.csv"
    
    env = DistributionNetworkEnv(
        node_file=node_file,
        line_file=line_file,
        time_series_file=time_series_file,
        time_step=0.25,  # 15分钟
        episode_length=672  # 7天 = 672步
    )
    
    print("\n环境创建成功!")
    print(f"  - 节点数: {env.n_nodes}")
    print(f"  - 状态维度: {env.state_dim}")
    print(f"  - 动作维度: {env.action_dim}")
    print(f"  - 时间步长: {env.time_step} 小时 (15分钟)")
    print(f"  - 回合长度: {env.episode_length} 步 (7天)")
    
    print(f"\n储能配置:")
    print(f"  - PSH: 节点34")
    print(f"    * 额定发电功率: 3.3 MW")
    print(f"    * 额定抽水功率: 3.3 MW")
    print(f"    * 上水库容量: 20.0 MWh")
    print(f"    * 下水库容量: 20.0 MWh")
    print(f"  - BESS1: 节点16, 1.3MW/2.6MWh")
    print(f"  - BESS2: 节点27, 1.3MW/2.6MWh")
    
    return env


def train():
    """训练PPO智能体"""
    print("\n" + "=" * 70)
    print("训练PPO智能体 - 版本4.9.2 (修复版)")
    print("=" * 70)
    
    set_random_seed(42)
    
    env = create_environment()
    
    # PPO配置 - 修复版
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=4,  # 4个离散动作: HOLD, GEN, PUMP, STOP
        lr=1e-5,  # 进一步降低学习率
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.1,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dims=[256, 256],  # 恢复较小的网络
        device='cpu'
    )
    
    print(f"\nPPO智能体配置:")
    print(f"  - 学习率: 1e-5")
    print(f"  - 折扣因子: 0.99")
    print(f"  - GAE lambda: 0.95")
    print(f"  - Clip epsilon: 0.1")
    print(f"  - 价值系数: 0.5")
    print(f"  - 熵系数: 0.01")
    print(f"  - 隐藏层: [256, 256]")
    print(f"  - 奖励缩放: 固定0.01")
    
    # 获取脚本所在目录，用于保存文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        max_episodes=1000,
        max_steps_per_episode=672,
        update_interval=2048,
        eval_interval=10,
        save_interval=20,
        log_interval=1,
        log_save_path=os.path.join(script_dir, "training_log.csv"),
        plot_save_path=os.path.join(script_dir, "training_plots.png"),
        max_constraint_violations=7,
        patience=30
    )
    
    print("\n开始训练...")
    trainer.train()
    
    # 保存最终模型
    model_path = os.path.join(script_dir, "ppo_model_final.pth")
    agent.save(model_path)
    print(f"\n最终模型已保存到 {model_path}")
    
    # 生成最终报告
    generate_final_report(trainer)
    
    return agent, env


def generate_final_report(trainer):
    """生成最终训练报告"""
    report = []
    report.append("\n" + "=" * 80)
    report.append("抽水储能与电池储能协同调度系统 - 版本4.9.2 (修复版) 最终报告")
    report.append("=" * 80)
    
    # 训练结果摘要
    report.append("\n## 训练结果摘要")
    report.append(f"\n### 基本信息")
    report.append(f"- 总训练轮次: {len(trainer.episode_rewards)}轮")
    report.append(f"- 每轮步数: 672步 (7天)")
    report.append(f"- 总训练步数: {len(trainer.episode_rewards) * 672:,}步")
    report.append(f"- 算法: PPO (Proximal Policy Optimization)")
    
    # PSH约束违反统计
    violation_list = trainer.episode_constraint_violations
    compliant_episodes = sum(1 for v in violation_list if v < trainer.max_constraint_violations)
    report.append(f"\n### PSH约束违反统计")
    report.append(f"- 达标轮次 (<7次): {compliant_episodes}轮")
    report.append(f"- 达标率: {compliant_episodes / len(violation_list) * 100:.1f}%")
    report.append(f"- 平均PSH约束违反: {np.mean(violation_list):.1f}次/轮")
    report.append(f"- 最少违反: {min(violation_list)}次")
    report.append(f"- 最多违反: {max(violation_list)}次")
    
    # 奖励统计
    rewards = trainer.episode_rewards
    valid_rewards = [r for r in rewards if not np.isnan(r) and not np.isinf(r)]
    report.append(f"\n### 奖励统计")
    if valid_rewards:
        report.append(f"- 平均奖励: {np.mean(valid_rewards):.2f}")
        report.append(f"- 最高奖励: {max(valid_rewards):.2f} (第{rewards.index(max(valid_rewards))+1}轮)")
        report.append(f"- 最低奖励: {min(valid_rewards):.2f} (第{rewards.index(min(valid_rewards))+1}轮)")
        report.append(f"- 标准差: {np.std(valid_rewards):.2f}")
        
        # 前20轮 vs 后20轮
        if len(valid_rewards) >= 40:
            early_mean = np.mean(valid_rewards[:20])
            late_mean = np.mean(valid_rewards[-20:])
            report.append(f"- 前20轮平均: {early_mean:.2f}")
            report.append(f"- 后20轮平均: {late_mean:.2f}")
            if early_mean != 0:
                report.append(f"- 改进幅度: {(late_mean - early_mean) / abs(early_mean) * 100:.1f}%")
    
    # PSH动作统计
    total_actions = sum(trainer.psh_action_counts.values())
    if total_actions > 0:
        report.append(f"\n### PSH动作统计 (总次数)")
        report.append(f"- 保持: {trainer.psh_action_counts[0]}次 ({trainer.psh_action_counts[0]/total_actions*100:.1f}%)")
        report.append(f"- 发电: {trainer.psh_action_counts[1]}次 ({trainer.psh_action_counts[1]/total_actions*100:.1f}%)")
        report.append(f"- 抽水: {trainer.psh_action_counts[2]}次 ({trainer.psh_action_counts[2]/total_actions*100:.1f}%)")
        report.append(f"- 停止: {trainer.psh_action_counts[3]}次 ({trainer.psh_action_counts[3]/total_actions*100:.1f}%)")
    
    # 电压统计
    report.append(f"\n### 电压统计")
    report.append(f"- 平均电压越限: {np.mean(trainer.episode_voltage_violations):.1f}次/轮")
    
    # 损失统计
    if len(trainer.agent.episode_actor_losses) > 0:
        actor_losses = [l for l in trainer.agent.episode_actor_losses if not np.isnan(l)]
        critic_losses = [l for l in trainer.agent.episode_critic_losses if not np.isnan(l)]
        if actor_losses and critic_losses:
            report.append(f"\n### 损失统计")
            report.append(f"- 最终Actor损失: {actor_losses[-1]:.6f}")
            report.append(f"- 最终Critic损失: {critic_losses[-1]:.6f}")
    
    # 关键改进
    report.append(f"\n## 版本4.9.2 (修复版) 关键改进")
    report.append(f"\n1. 修复NaN损失问题")
    report.append(f"   - 进一步降低学习率: 3e-5 -> 1e-5")
    report.append(f"   - 禁用动态奖励归一化")
    report.append(f"   - 添加更严格的NaN检查")
    report.append(f"\n2. 修复损失值记录问题")
    report.append(f"   - 每轮都记录Actor/Critic损失")
    report.append(f"   - 增强可视化，包含所有损失曲线")
    report.append(f"\n3. 优化网络架构")
    report.append(f"   - 恢复隐藏层: [256, 256]")
    report.append(f"   - 使用更保守的网络初始化")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n最终报告已保存到 {report_path}")


def evaluate(model_path: str, num_episodes: int = 10):
    """评估训练好的模型"""
    print("\n" + "=" * 70)
    print("评估模型")
    print("=" * 70)
    
    env = create_environment()
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=4,  # PSH有4个离散动作: HOLD, GEN, PUMP, STOP
        hidden_dims=[256, 256],
        device='cpu'
    )
    
    print(f"加载模型: {model_path}")
    agent.load(model_path)
    
    print(f"\n评估 {num_episodes} 个回合...")
    
    all_rewards = []
    all_violations = []
    all_constraint_violations = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward_sum = 0.0
        episode_steps = 0
        episode_violations = 0
        episode_constraint_violations = 0
        
        for step in range(env.episode_length):
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
            
            episode_reward_sum += reward
            episode_steps += 1
            episode_violations += info.get('v_violation_count', 0)
            if info.get('psh_constraint_violated', False):
                episode_constraint_violations += 1
            
            state = next_state
            if done:
                break
        
        avg_reward = episode_reward_sum / max(episode_steps, 1)
        all_rewards.append(avg_reward)
        all_violations.append(episode_violations)
        all_constraint_violations.append(episode_constraint_violations)
        
        print(f"  Episode {episode + 1}: 平均奖励 = {avg_reward:.4f}, 电压越限 = {episode_violations}, PSH约束违反 = {episode_constraint_violations}")
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_violations = np.mean(all_violations)
    mean_constraint_violations = np.mean(all_constraint_violations)
    
    print(f"\n评估结果:")
    print(f"  平均奖励: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  平均电压越限次数: {mean_violations:.2f}")
    print(f"  平均PSH约束违反次数: {mean_constraint_violations:.2f}")
    
    return all_rewards, all_violations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='抽水储能与电池储能协同调度 - 版本4.9.2 (修复版)')
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['train', 'eval', 'all'],
        help='运行模式'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='ppo_model_final.pth',
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
    print("抽水储能与电池储能协同调度 - 版本4.9.2 (修复版)")
    print("=" * 70)
    print("\n重大改进:")
    print("  1. 修复NaN损失问题")
    print("  2. 降低学习率: 3e-5 -> 1e-5")
    print("  3. 禁用动态奖励归一化")
    print("  4. 添加更严格的NaN检查")
    print("  5. 增强可视化 - 包含Actor/Critic损失")
    print("=" * 70)
    
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        # 如果模型路径是相对路径，转换为绝对路径
        if not os.path.isabs(args.model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_path)
        else:
            model_path = args.model_path
        evaluate(model_path, args.num_eval_episodes)
    elif args.mode == 'all':
        agent, env = train()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_model_final.pth")
        evaluate(model_path, args.num_eval_episodes)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
