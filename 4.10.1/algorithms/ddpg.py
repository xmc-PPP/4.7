"""
Deep Deterministic Policy Gradient (DDPG) 算法实现 - 版本4.8.6

优化说明:
1. 添加训练监控和自动终止机制
2. 改进网络结构
3. 添加早停机制
4. 监控PSH约束违反次数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import copy
import random
from collections import deque
import pandas as pd
import os


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # 检查输入有效性
        if np.isnan(state).any() or np.isnan(action).any() or np.isnan(reward) or np.isnan(next_state).any():
            return  # 跳过无效数据
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor网络 - 改进版本"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: List[int] = [256, 256],
        init_w: float = 3e-3
    ):
        super(Actor, self).__init__()
        
        self.layers = nn.ModuleList()
        
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        action = torch.tanh(self.output_layer(x))
        return action


class Critic(nn.Module):
    """Critic网络 - 改进版本"""
    
    def __init__(
            self,
            state_dim: int,
            action_dim: int = 3,
            hidden_dims: List[int] = [256, 256],
            init_w: float = 3e-3
    ):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        
        self.layers = nn.ModuleList()
        
        prev_dim = hidden_dims[0] + action_dim
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0)
        
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        
        for layer in self.layers:
            x = F.relu(layer(x))
        
        q_value = self.output_layer(x)
        return q_value


class DDPGAgent:
    """DDPG智能体 - 改进版本"""
    
    def __init__(
            self,
            state_dim: int,
            action_dim: int = 3,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            buffer_capacity: int = 100000,
            batch_size: int = 128,
            hidden_dims: List[int] = [256, 256],
            warmup_steps: int = 2000,
            device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 噪声参数 - 版本4.8.7: 降低噪声，更快收敛
        self.noise_std = 0.2  # 降低初始噪声
        self.noise_decay = 0.995  # 加快衰减
        self.min_noise_std = 0.02  # 降低最小噪声
        
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        self.q_values_target = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作"""
        # 检查状态有效性
        if np.isnan(state).any():
            return np.zeros(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
        
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def update(self) -> Dict:
        """更新网络"""
        if self.total_steps < self.warmup_steps:
            self.total_steps += 1
            return {}
        
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 检查tensor有效性
        if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(rewards).any():
            return {}
        
        # 裁剪奖励
        rewards = torch.clamp(rewards, -15, 10)
        
        # Critic更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            target_q = torch.clamp(target_q, -50, 50)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 检查critic_loss有效性
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            return {}
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor更新
        actor_loss = torch.tensor(0.0)
        if self.total_steps % 2 == 0:
            pred_actions = self.actor(states)
            actor_loss = -self.critic(states, pred_actions).mean()
            
            # 检查actor_loss有效性
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                return {}
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        self.total_steps += 1
        
        # 记录损失
        self.actor_losses.append(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0)
        self.critic_losses.append(critic_loss.item())
        self.q_values.append(current_q.mean().item())
        self.q_values_target.append(target_q.mean().item())
        
        return {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0,
            'critic_loss': critic_loss.item(),
            'q_value': current_q.mean().item(),
            'q_target': target_q.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def decay_noise(self):
        """衰减探索噪声"""
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_std': self.noise_std,
            'total_steps': self.total_steps
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.noise_std = checkpoint.get('noise_std', 0.1)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


class DDPGTrainer:
    """DDPG训练器 - 版本4.8.6，添加训练监控和自动终止机制"""
    
    def __init__(
            self,
            env,
            agent: DDPGAgent,
            max_episodes: int = 200,
            max_steps_per_episode: int = 672,
            eval_interval: int = 10,
            save_interval: int = 20,
            log_interval: int = 1,
            log_save_path: str = "training_log.csv",
            plot_save_path: str = "training_plots.png",
            # === 版本4.8.6: 添加监控参数 ===
            max_constraint_violations: int = 7,  # 最大允许的PSH约束违反次数
            patience: int = 20,  # 早停耐心值
            min_improvement: float = 0.01  # 最小改进阈值
    ):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        self.log_save_path = log_save_path
        self.plot_save_path = plot_save_path
        
        # === 版本4.8.6: 监控参数 ===
        self.max_constraint_violations = max_constraint_violations
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.episode_voltage_violations = []
        self.episode_soc_stats = []
        self.episode_constraint_violations = []
        self.psh_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        self.detailed_logs = []
        
        # 早停相关
        self.best_eval_reward = -np.inf
        self.patience_counter = 0
        self.best_constraint_violations = float('inf')
        
        # 训练状态
        self.should_stop = False
        self.stop_reason = ""
    
    def train(self):
        """训练智能体 - 版本4.8.6，添加监控和自动终止"""
        print("=" * 70)
        print("开始训练...")
        print(f"训练设置: {self.max_episodes}轮, 每轮{self.max_steps_per_episode}步(7天)")
        print(f"PSH约束违反目标: <{self.max_constraint_violations}次/轮")
        print("=" * 70)
        
        for episode in range(self.max_episodes):
            if self.should_stop:
                print(f"\n{'='*70}")
                print(f"训练提前终止: {self.stop_reason}")
                print(f"{'='*70}")
                break
            
            # 重置环境
            state = self.env.reset()
            episode_reward_sum = 0.0
            episode_steps = 0
            
            episode_violations = 0
            episode_constraint_violations = 0
            psh_powers = []
            bess1_powers = []
            bess2_powers = []
            psh_upper_socs = []
            psh_lower_socs = []
            bess1_socs = []
            bess2_socs = []
            voltages_min = []
            voltages_max = []
            episode_psh_actions = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state, add_noise=True)
                next_state, reward, done, info = self.env.step(action)
                
                # 检查奖励有效性
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                update_info = self.agent.update()
                
                episode_reward_sum += reward
                episode_steps += 1
                
                episode_violations += info.get('v_violation_count', 0)
                
                if info.get('psh_constraint_violated', False):
                    episode_constraint_violations += 1
                
                if 'psh_action' in info:
                    episode_psh_actions[info['psh_action']] += 1
                
                if 'voltage' in info:
                    voltages_min.append(info['voltage_min'])
                    voltages_max.append(info['voltage_max'])
                
                psh_powers.append(info.get('psh_power', 0))
                bess1_powers.append(info['bess_powers'][0] if 'bess_powers' in info else 0)
                bess2_powers.append(info['bess_powers'][1] if 'bess_powers' in info else 0)
                psh_upper_socs.append(info.get('psh_upper_soc', 0.5))
                psh_lower_socs.append(info.get('psh_lower_soc', 0.5))
                bess1_socs.append(info['bess_socs'][0] if 'bess_socs' in info else 0.5)
                bess2_socs.append(info['bess_socs'][1] if 'bess_socs' in info else 0.5)
                
                state = next_state
                if done:
                    break
            
            # 衰减噪声
            self.agent.decay_noise()
            
            # 计算平均奖励
            avg_episode_reward = episode_reward_sum / max(episode_steps, 1)
            
            self.episode_rewards.append(avg_episode_reward)
            self.episode_lengths.append(episode_steps)
            self.episode_voltage_violations.append(episode_violations)
            self.episode_constraint_violations.append(episode_constraint_violations)
            
            for k, v in episode_psh_actions.items():
                self.psh_action_counts[k] += v
            
            soc_stats = {
                'psh_upper_mean': np.mean(psh_upper_socs) if psh_upper_socs else 0.5,
                'psh_upper_min': np.min(psh_upper_socs) if psh_upper_socs else 0.5,
                'psh_upper_max': np.max(psh_upper_socs) if psh_upper_socs else 0.5,
                'psh_lower_mean': np.mean(psh_lower_socs) if psh_lower_socs else 0.5,
                'psh_lower_min': np.min(psh_lower_socs) if psh_lower_socs else 0.5,
                'psh_lower_max': np.max(psh_lower_socs) if psh_lower_socs else 0.5,
                'bess1_mean': np.mean(bess1_socs) if bess1_socs else 0.5,
                'bess2_mean': np.mean(bess2_socs) if bess2_socs else 0.5
            }
            self.episode_soc_stats.append(soc_stats)
            
            # 计算PSH动作统计
            psh_hold_count = episode_psh_actions[0]
            psh_gen_count = episode_psh_actions[1]
            psh_pump_count = episode_psh_actions[2]
            psh_stop_count = episode_psh_actions[3]
            
            # 每轮输出日志
            if (episode + 1) % self.log_interval == 0:
                print(f"\n{'='*70}")
                print(f"训练轮次 {episode + 1}/{self.max_episodes}")
                print(f"{'='*70}")
                print(f"平均奖励: {avg_episode_reward:.4f}")
                print(f"噪声标准差: {self.agent.noise_std:.4f}")
                print(f"电压越限次数: {episode_violations}")
                print(f"PSH约束违反: {episode_constraint_violations}")
                print(f"PSH动作: 保持={psh_hold_count}, 发电={psh_gen_count}, 抽水={psh_pump_count}, 停止={psh_stop_count}")
                print(f"PSH上水库SOC: {soc_stats['psh_upper_mean']:.3f} (范围: {soc_stats['psh_upper_min']:.3f}-{soc_stats['psh_upper_max']:.3f})")
                print(f"PSH下水库SOC: {soc_stats['psh_lower_mean']:.3f} (范围: {soc_stats['psh_lower_min']:.3f}-{soc_stats['psh_lower_max']:.3f})")
                if voltages_min:
                    print(f"电压范围: [{min(voltages_min):.4f}, {max(voltages_max):.4f}]")
                
                # === 版本4.8.6: 检查是否达到目标 ===
                if episode_constraint_violations < self.max_constraint_violations:
                    print(f">>> PSH约束违反达标! ({episode_constraint_violations} < {self.max_constraint_violations})")
            
            # 保存详细日志
            log_entry = {
                '训练轮次': episode + 1,
                '平均奖励': round(avg_episode_reward, 4),
                '噪声标准差': round(self.agent.noise_std, 4),
                '电压越限次数': episode_violations,
                'PSH约束违反': episode_constraint_violations,
                'PSH保持次数': psh_hold_count,
                'PSH发电次数': psh_gen_count,
                'PSH抽水次数': psh_pump_count,
                'PSH停止次数': psh_stop_count,
                '上水库SOC': round(soc_stats['psh_upper_mean'], 3),
                '下水库SOC': round(soc_stats['psh_lower_mean'], 3),
                '电压最小值': round(min(voltages_min), 4) if voltages_min else 0.95,
                '电压最大值': round(max(voltages_max), 4) if voltages_max else 1.05,
                '评估奖励': ''
            }
            self.detailed_logs.append(log_entry)
            
            # 每轮保存日志
            self._save_training_log()
            
            # === 版本4.8.6: 监控和自动终止检查 ===
            self._check_training_status(episode, episode_constraint_violations)
            
            # 评估
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                
                if self.detailed_logs:
                    self.detailed_logs[-1]['评估奖励'] = round(eval_reward, 4)
                    self._save_training_log()
                
                print(f"\n>>> 第 {episode + 1} 轮评估平均奖励: {eval_reward:.4f}")
                
                # 更新最佳评估奖励
                if eval_reward > self.best_eval_reward + self.min_improvement:
                    self.best_eval_reward = eval_reward
                    self.patience_counter = 0
                    self.agent.save("best_model.pth")
                    print(">>> 保存最佳模型")
                else:
                    self.patience_counter += 1
                
                # 学习率调整
                if self.patience_counter >= self.patience:
                    for param_group in self.agent.actor_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    for param_group in self.agent.critic_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(">>> 学习率降低50%")
                    self.patience_counter = 0
            
            # 保存检查点
            if (episode + 1) % self.save_interval == 0:
                self.agent.save(f"ddpg_checkpoint_episode_{episode + 1}.pth")
        
        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        
        # 最终保存
        self._save_training_log()
        self.plot_training_history()
    
    def _check_training_status(self, episode: int, constraint_violations: int):
        """
        检查训练状态 - 版本4.8.6
        监控指标并向不好方向发展时触发终止
        """
        # 检查PSH约束违反次数是否持续恶化
        if len(self.episode_constraint_violations) >= 10:
            recent_violations = self.episode_constraint_violations[-10:]
            avg_recent = np.mean(recent_violations)
            
            # 如果最近10轮的平均违反次数超过目标的两倍，触发终止
            if avg_recent > self.max_constraint_violations * 2 and episode > 50:
                self.should_stop = True
                self.stop_reason = f"PSH约束违反持续恶化 (最近10轮平均: {avg_recent:.1f})"
                return
        
        # 检查奖励是否持续下降
        if len(self.episode_rewards) >= 20:
            recent_rewards = self.episode_rewards[-20:]
            early_rewards = self.episode_rewards[-40:-20] if len(self.episode_rewards) >= 40 else self.episode_rewards[:20]
            
            if len(early_rewards) > 0:
                avg_recent = np.mean(recent_rewards)
                avg_early = np.mean(early_rewards)
                
                # 如果奖励下降超过50%，触发终止
                if avg_recent < avg_early * 0.5 and episode > 100:
                    self.should_stop = True
                    self.stop_reason = f"奖励持续下降 (近期: {avg_recent:.4f}, 早期: {avg_early:.4f})"
                    return
        
        # 检查是否连续多轮达到目标
        if len(self.episode_constraint_violations) >= 20:
            recent_violations = self.episode_constraint_violations[-20:]
            达标次数 = sum(1 for v in recent_violations if v < self.max_constraint_violations)
            
            if 达标次数 >= 15:  # 20轮中有15轮达标
                self.should_stop = True
                self.stop_reason = f"训练已收敛 (最近20轮有{达标次数}轮PSH约束违反达标)"
                return
    
    def _save_training_log(self):
        """保存训练日志到CSV文件"""
        try:
            df = pd.DataFrame(self.detailed_logs)
            columns = ['训练轮次', '平均奖励', '噪声标准差', '电压越限次数', 'PSH约束违反',
                      'PSH保持次数', 'PSH发电次数', 'PSH抽水次数', 'PSH停止次数',
                      '上水库SOC', '下水库SOC', '电压最小值', '电压最大值', '评估奖励']
            
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
            
            df = df[columns]
            df.to_csv(self.log_save_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存日志出错: {e}")
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """评估智能体"""
        total_reward = 0.0
        total_steps = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward_sum = 0.0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state, add_noise=False)
                next_state, reward, done, info = self.env.step(action)
                
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                episode_reward_sum += reward
                episode_steps += 1
                
                state = next_state
                if done:
                    break
            
            total_reward += episode_reward_sum
            total_steps += episode_steps
        
        # 返回平均奖励
        avg_reward = total_reward / max(total_steps, 1)
        return avg_reward
    
    def plot_training_history(self):
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # 平均奖励
            axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='平均奖励')
            if len(self.episode_rewards) >= 10:
                ma = np.convolve(self.episode_rewards, np.ones(10) / 10, mode='valid')
                axes[0, 0].plot(range(9, len(self.episode_rewards)), ma, 'r-', linewidth=2, label='MA(10)')
            axes[0, 0].set_title('每轮平均奖励')
            axes[0, 0].set_xlabel('训练轮次')
            axes[0, 0].set_ylabel('平均奖励')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 评估奖励
            if len(self.eval_rewards) > 0:
                eval_episodes = range(self.eval_interval, len(self.eval_rewards) * self.eval_interval + 1,
                                      self.eval_interval)
                axes[0, 1].plot(eval_episodes, self.eval_rewards, 'bo-', markersize=4)
                axes[0, 1].set_title('评估平均奖励')
                axes[0, 1].set_xlabel('训练轮次')
                axes[0, 1].set_ylabel('平均奖励')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 电压越限次数
            axes[0, 2].plot(self.episode_voltage_violations, alpha=0.6)
            axes[0, 2].set_title('每轮电压越限次数')
            axes[0, 2].set_xlabel('训练轮次')
            axes[0, 2].set_ylabel('越限次数')
            axes[0, 2].grid(True, alpha=0.3)
            
            # PSH约束违反
            axes[1, 0].plot(self.episode_constraint_violations, alpha=0.6, color='orange')
            # 添加目标线
            axes[1, 0].axhline(y=self.max_constraint_violations, color='r', linestyle='--', 
                              label=f'目标: {self.max_constraint_violations}')
            axes[1, 0].set_title('每轮PSH约束违反次数')
            axes[1, 0].set_xlabel('训练轮次')
            axes[1, 0].set_ylabel('违反次数')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 网络损失
            if len(self.agent.actor_losses) > 0 and len(self.agent.critic_losses) > 0:
                axes[1, 1].plot(self.agent.actor_losses, alpha=0.6, label='Actor')
                axes[1, 1].plot(self.agent.critic_losses, alpha=0.6, label='Critic')
                axes[1, 1].set_title('网络损失')
                axes[1, 1].set_xlabel('更新步数')
                axes[1, 1].set_ylabel('损失')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Q值
            if len(self.agent.q_values) > 0:
                axes[1, 2].plot(self.agent.q_values, alpha=0.6, label='当前Q值')
                if len(self.agent.q_values_target) > 0:
                    axes[1, 2].plot(self.agent.q_values_target, alpha=0.6, label='目标Q值')
                axes[1, 2].set_title('Q值')
                axes[1, 2].set_xlabel('更新步数')
                axes[1, 2].set_ylabel('Q值')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plot_save_path, dpi=150, bbox_inches='tight')
            print(f"训练可视化已保存到: {self.plot_save_path}")
            plt.close()
        except Exception as e:
            print(f"绘制可视化出错: {e}")
