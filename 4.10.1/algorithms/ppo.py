"""
Proximal Policy Optimization (PPO) 算法实现 - 版本4.9.2 (修复版)

优化改进:
1. 修复损失值记录问题 - 每轮都记录
2. 降低学习率，提高训练稳定性
3. 禁用奖励归一化（避免NaN）
4. 添加更严格的NaN检查
5. 优化网络初始化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from torch.distributions import Categorical
import copy


class ActorNetwork(nn.Module):
    """Actor网络 - 输出动作概率"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        
        # 初始化 - 使用更保守的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        action_logits = self.action_head(features)
        return action_logits
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        action_logits = self.forward(state)
        
        # 检查NaN
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            probs = torch.ones_like(action_logits) / action_logits.shape[-1]
        else:
            probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
        
        return action, probs
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作"""
        action_logits = self.forward(state)
        
        # 检查NaN
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            batch_size = action_logits.shape[0]
            action_dim = action_logits.shape[-1]
            log_probs = torch.zeros(batch_size, device=action_logits.device)
            entropy = torch.log(torch.tensor(float(action_dim))) * torch.ones(batch_size, device=action_logits.device)
            return log_probs, entropy
        
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Critic网络 - 输出状态价值"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        value = self.value_head(features)
        return value


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO智能体 - 版本4.9.2 (修复版)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-5,  # 进一步降低学习率
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5  # 增加数值稳定性
        )
        
        self.buffer = RolloutBuffer()
        
        # 记录损失 - 每轮都记录
        self.episode_actor_losses = []
        self.episode_critic_losses = []
        self.episode_entropy_losses = []
        self.episode_total_losses = []
        
        # 当前回合的损失累加
        self.current_actor_loss = 0.0
        self.current_critic_loss = 0.0
        self.current_entropy_loss = 0.0
        self.current_total_loss = 0.0
        self.current_update_count = 0
        
        # 奖励缩放参数
        self.reward_scale = 0.01  # 固定奖励缩放
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            # 检查NaN
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                action_dim = action_logits.shape[-1]
                probs = torch.ones(1, action_dim) / action_dim
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = torch.randint(0, action_dim, (1,))
                log_prob = 0.0
            else:
                probs = F.softmax(action_logits, dim=-1)
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                    log_prob = 0.0
                else:
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).item()
        
        return action.item(), log_prob, value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算GAE优势"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_state: np.ndarray, n_epochs: int = 4, batch_size: int = 64):
        """更新网络"""
        if len(self.buffer) == 0:
            return {}
        
        # 获取数据
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # 奖励缩放（固定值，避免动态归一化的问题）
        rewards = rewards * self.reward_scale
        
        # 检查奖励是否有NaN
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print("警告: 奖励中有NaN或Inf，跳过本次更新")
            self.buffer.clear()
            return {}
        
        # 计算下一个状态的价值
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).item()
        
        # 计算GAE优势
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 检查优势是否有NaN
        if np.isnan(advantages).any() or np.isinf(advantages).any():
            print("警告: 优势中有NaN或Inf，跳过本次更新")
            self.buffer.clear()
            return {}
        
        # 转换为tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        if len(advantages) > 1:
            adv_mean = advantages_tensor.mean()
            adv_std = advantages_tensor.std()
            if adv_std > 1e-8:
                advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
        
        # 多轮更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        nan_detected = False
        
        for epoch in range(n_epochs):
            if nan_detected:
                break
                
            # 随机打乱数据
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 评估动作
                log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                values_pred = self.critic(batch_states).squeeze(-1)
                
                # 检查NaN
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    print(f"警告: Epoch {epoch} 检测到NaN，跳过")
                    nan_detected = True
                    break
                
                if torch.isnan(values_pred).any() or torch.isinf(values_pred).any():
                    print(f"警告: Epoch {epoch} 价值预测中有NaN，跳过")
                    nan_detected = True
                    break
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 检查比率
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print(f"警告: Epoch {epoch} 比率中有NaN，跳过")
                    nan_detected = True
                    break
                
                # 计算裁剪后的目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                critic_loss = F.mse_loss(values_pred, batch_returns)
                
                # 计算熵损失
                entropy_loss = -entropy.mean()
                
                # 检查损失
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    print(f"警告: Epoch {epoch} 损失为NaN，跳过")
                    nan_detected = True
                    break
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 记录损失
        if n_updates > 0 and not nan_detected:
            avg_actor_loss = total_actor_loss / n_updates
            avg_critic_loss = total_critic_loss / n_updates
            avg_entropy_loss = total_entropy_loss / n_updates
            avg_total_loss = avg_actor_loss + self.value_coef * avg_critic_loss + self.entropy_coef * avg_entropy_loss
            
            # 累加到当前回合
            self.current_actor_loss += avg_actor_loss
            self.current_critic_loss += avg_critic_loss
            self.current_entropy_loss += avg_entropy_loss
            self.current_total_loss += avg_total_loss
            self.current_update_count += 1
        
        return {
            'actor_loss': self.current_actor_loss / max(self.current_update_count, 1),
            'critic_loss': self.current_critic_loss / max(self.current_update_count, 1),
            'entropy_loss': self.current_entropy_loss / max(self.current_update_count, 1),
            'total_loss': self.current_total_loss / max(self.current_update_count, 1)
        }
    
    def end_episode(self):
        """结束回合 - 记录本回合的平均损失"""
        if self.current_update_count > 0:
            self.episode_actor_losses.append(self.current_actor_loss / self.current_update_count)
            self.episode_critic_losses.append(self.current_critic_loss / self.current_update_count)
            self.episode_entropy_losses.append(self.current_entropy_loss / self.current_update_count)
            self.episode_total_losses.append(self.current_total_loss / self.current_update_count)
        else:
            # 如果没有更新，使用上一个回合的损失或0
            if len(self.episode_actor_losses) > 0:
                self.episode_actor_losses.append(self.episode_actor_losses[-1])
                self.episode_critic_losses.append(self.episode_critic_losses[-1])
                self.episode_entropy_losses.append(self.episode_entropy_losses[-1])
                self.episode_total_losses.append(self.episode_total_losses[-1])
            else:
                self.episode_actor_losses.append(0.0)
                self.episode_critic_losses.append(0.0)
                self.episode_entropy_losses.append(0.0)
                self.episode_total_losses.append(0.0)
        
        # 重置当前回合累加器
        self.current_actor_loss = 0.0
        self.current_critic_loss = 0.0
        self.current_entropy_loss = 0.0
        self.current_total_loss = 0.0
        self.current_update_count = 0
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_actor_losses': self.episode_actor_losses,
            'episode_critic_losses': self.episode_critic_losses,
            'episode_entropy_losses': self.episode_entropy_losses,
            'episode_total_losses': self.episode_total_losses,
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class PPOTrainer:
    """PPO训练器 - 版本4.9.2 (修复版)"""
    
    def __init__(
        self,
        env,
        agent: PPOAgent,
        max_episodes: int = 200,
        max_steps_per_episode: int = 672,
        update_interval: int = 2048,
        eval_interval: int = 10,
        save_interval: int = 20,
        log_interval: int = 1,
        log_save_path: str = "training_log.csv",
        plot_save_path: str = "training_plots.png",
        max_constraint_violations: int = 7,
        patience: int = 30
    ):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        self.log_save_path = log_save_path
        self.plot_save_path = plot_save_path
        
        self.max_constraint_violations = max_constraint_violations
        self.patience = patience
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.episode_voltage_violations = []
        self.episode_constraint_violations = []
        self.psh_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        self.detailed_logs = []
        
        self.best_eval_reward = -np.inf
        self.patience_counter = 0
        
        self.should_stop = False
        self.stop_reason = ""
    
    def train(self):
        """训练智能体"""
        print("=" * 70)
        print("PPO训练开始 - 版本4.9.2 (修复版)")
        print(f"训练设置: {self.max_episodes}轮, 每轮{self.max_steps_per_episode}步")
        print(f"PSH约束违反目标: <{self.max_constraint_violations}次/轮")
        print("=" * 70)
        
        total_steps = 0
        
        for episode in range(self.max_episodes):
            if self.should_stop:
                print(f"\n{'='*70}")
                print(f"训练提前终止: {self.stop_reason}")
                print(f"{'='*70}")
                break
            
            state = self.env.reset()
            episode_reward_sum = 0.0
            episode_steps = 0
            
            episode_violations = 0
            episode_constraint_violations = 0
            psh_powers = []
            psh_upper_socs = []
            psh_lower_socs = []
            voltages_min = []
            voltages_max = []
            episode_psh_actions = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for step in range(self.max_steps_per_episode):
                action, log_prob, value = self.agent.select_action(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                
                # 检查奖励
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                # 限制奖励范围
                reward = np.clip(reward, -100, 100)
                
                # 存储经验
                self.agent.buffer.push(state, action, reward, value, log_prob, done)
                
                episode_reward_sum += reward
                episode_steps += 1
                total_steps += 1
                
                episode_violations += info.get('v_violation_count', 0)
                
                if info.get('psh_constraint_violated', False):
                    episode_constraint_violations += 1
                
                if 'psh_action' in info:
                    episode_psh_actions[info['psh_action']] += 1
                
                if 'voltage' in info:
                    voltages_min.append(info['voltage_min'])
                    voltages_max.append(info['voltage_max'])
                
                psh_powers.append(info.get('psh_power', 0))
                psh_upper_socs.append(info.get('psh_upper_soc', 0.5))
                psh_lower_socs.append(info.get('psh_lower_soc', 0.5))
                
                state = next_state
                
                # 定期更新
                if len(self.agent.buffer) >= self.update_interval:
                    update_info = self.agent.update(next_state)
                
                if done:
                    break
            
            # 如果缓冲区还有数据，更新
            if len(self.agent.buffer) > 0:
                update_info = self.agent.update(next_state)
            
            # 结束回合 - 记录损失
            self.agent.end_episode()
            
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
            }
            
            psh_hold_count = episode_psh_actions[0]
            psh_gen_count = episode_psh_actions[1]
            psh_pump_count = episode_psh_actions[2]
            psh_stop_count = episode_psh_actions[3]
            
            # 输出日志
            if (episode + 1) % self.log_interval == 0:
                print(f"\n{'='*70}")
                print(f"训练轮次 {episode + 1}/{self.max_episodes}")
                print(f"{'='*70}")
                print(f"平均奖励: {avg_episode_reward:.4f}")
                print(f"电压越限次数: {episode_violations}")
                print(f"PSH约束违反: {episode_constraint_violations}")
                print(f"PSH动作: 保持={psh_hold_count}, 发电={psh_gen_count}, 抽水={psh_pump_count}, 停止={psh_stop_count}")
                print(f"PSH上水库SOC: {soc_stats['psh_upper_mean']:.3f} (范围: {soc_stats['psh_upper_min']:.3f}-{soc_stats['psh_upper_max']:.3f})")
                if voltages_min:
                    print(f"电压范围: [{min(voltages_min):.4f}, {max(voltages_max):.4f}]")
                
                # 打印损失
                if len(self.agent.episode_actor_losses) > 0:
                    actor_loss = self.agent.episode_actor_losses[-1]
                    critic_loss = self.agent.episode_critic_losses[-1]
                    print(f"Actor损失: {actor_loss:.6f}" if not np.isnan(actor_loss) else "Actor损失: NaN")
                    print(f"Critic损失: {critic_loss:.6f}" if not np.isnan(critic_loss) else "Critic损失: NaN")
                
                if episode_constraint_violations < self.max_constraint_violations:
                    print(f">>> PSH约束违反达标! ({episode_constraint_violations} < {self.max_constraint_violations})")
            
            # 保存日志
            log_entry = {
                '训练轮次': episode + 1,
                '平均奖励': round(avg_episode_reward, 4),
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
                '总损失': round(self.agent.episode_total_losses[-1], 6) if len(self.agent.episode_total_losses) > 0 else 0,
                'Actor损失': round(self.agent.episode_actor_losses[-1], 6) if len(self.agent.episode_actor_losses) > 0 else 0,
                'Critic损失': round(self.agent.episode_critic_losses[-1], 6) if len(self.agent.episode_critic_losses) > 0 else 0,
                '评估奖励': ''
            }
            
            self.detailed_logs.append(log_entry)
            self._save_training_log()
            
            # 检查训练状态
            self._check_training_status(episode, episode_constraint_violations)
            
            # 评估
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                
                if self.detailed_logs:
                    self.detailed_logs[-1]['评估奖励'] = round(eval_reward, 4)
                    self._save_training_log()
                
                print(f"\n>>> 第 {episode + 1} 轮评估平均奖励: {eval_reward:.4f}")
                
                # 保存最佳模型
                if eval_reward > self.best_eval_reward + 0.001:
                    self.best_eval_reward = eval_reward
                    self.patience_counter = 0
                    self.agent.save("best_model.pth")
                    print(">>> 保存最佳模型")
                else:
                    self.patience_counter += 1
                    print(f">>> 未改进 ({self.patience_counter}/{self.patience})")
                
                # 早停检查
                if self.patience_counter >= self.patience:
                    print(">>> 触发早停")
                    self.should_stop = True
                    self.stop_reason = "评估奖励长期未改进"
            
            # 保存检查点
            if (episode + 1) % self.save_interval == 0:
                self.agent.save(f"ppo_checkpoint_episode_{episode + 1}.pth")
        
        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        
        self._save_training_log()
        self.plot_training_history()
    
    def _check_training_status(self, episode: int, constraint_violations: int):
        """检查训练状态"""
        # 检查PSH约束违反
        if len(self.episode_constraint_violations) >= 15:
            recent_violations = self.episode_constraint_violations[-15:]
            avg_recent = np.mean(recent_violations)
            
            if avg_recent > self.max_constraint_violations * 3 and episode > 30:
                self.should_stop = True
                self.stop_reason = f"PSH约束违反持续恶化 (最近15轮平均: {avg_recent:.1f})"
                return
        
        # 检查奖励是否有NaN
        if len(self.episode_rewards) >= 5:
            recent_rewards = self.episode_rewards[-5:]
            if any(np.isnan(r) or np.isinf(r) for r in recent_rewards):
                self.should_stop = True
                self.stop_reason = "检测到NaN或Inf奖励"
                return
        
        # 检查损失是否有NaN
        if len(self.agent.episode_actor_losses) >= 5:
            recent_losses = self.agent.episode_actor_losses[-5:]
            if any(np.isnan(l) or np.isinf(l) for l in recent_losses):
                self.should_stop = True
                self.stop_reason = "检测到NaN或Inf损失"
                return
    
    def _save_training_log(self):
        """保存训练日志"""
        try:
            import pandas as pd
            df = pd.DataFrame(self.detailed_logs)
            columns = ['训练轮次', '平均奖励', '电压越限次数', 'PSH约束违反',
                      'PSH保持次数', 'PSH发电次数', 'PSH抽水次数', 'PSH停止次数',
                      '上水库SOC', '下水库SOC', '电压最小值', '电压最大值',
                      '总损失', 'Actor损失', 'Critic损失', '评估奖励']
            
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
            
            df = df[columns]
            df.to_csv(self.log_save_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存日志出错: {e}")
    
    def evaluate(self, num_episodes: int = 3) -> float:
        """评估智能体"""
        total_reward = 0.0
        total_steps = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward_sum = 0.0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                action, _, _ = self.agent.select_action(state, deterministic=True)
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
        
        avg_reward = total_reward / max(total_steps, 1)
        return avg_reward
    
    def plot_training_history(self):
        """绘制训练历史 - 增强版"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置中文字体支持
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            
            # 1. 平均奖励
            axes[0, 0].plot(self.episode_rewards, alpha=0.5, color='blue', label='原始')
            if len(self.episode_rewards) >= 10:
                ma10 = np.convolve(self.episode_rewards, np.ones(10) / 10, mode='valid')
                axes[0, 0].plot(range(9, len(self.episode_rewards)), ma10, 'r-', linewidth=2, label='MA(10)')
            if len(self.episode_rewards) >= 30:
                ma30 = np.convolve(self.episode_rewards, np.ones(30) / 30, mode='valid')
                axes[0, 0].plot(range(29, len(self.episode_rewards)), ma30, 'g-', linewidth=2, label='MA(30)')
            axes[0, 0].set_title('每轮平均奖励', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('训练轮次')
            axes[0, 0].set_ylabel('平均奖励')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 评估奖励
            if len(self.eval_rewards) > 0:
                eval_episodes = range(self.eval_interval, len(self.eval_rewards) * self.eval_interval + 1,
                                      self.eval_interval)
                axes[0, 1].plot(eval_episodes, self.eval_rewards, 'bo-', markersize=6, linewidth=2)
                axes[0, 1].set_title('评估平均奖励', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('训练轮次')
                axes[0, 1].set_ylabel('平均奖励')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 电压越限次数
            axes[0, 2].plot(self.episode_voltage_violations, alpha=0.6, color='purple')
            if len(self.episode_voltage_violations) >= 10:
                ma = np.convolve(self.episode_voltage_violations, np.ones(10) / 10, mode='valid')
                axes[0, 2].plot(range(9, len(self.episode_voltage_violations)), ma, 'r-', linewidth=2)
            axes[0, 2].set_title('每轮电压越限次数', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('训练轮次')
            axes[0, 2].set_ylabel('越限次数')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. PSH约束违反
            axes[1, 0].plot(self.episode_constraint_violations, alpha=0.6, color='orange')
            axes[1, 0].axhline(y=self.max_constraint_violations, color='r', linestyle='--', 
                              linewidth=2, label=f'目标: {self.max_constraint_violations}')
            if len(self.episode_constraint_violations) >= 10:
                ma = np.convolve(self.episode_constraint_violations, np.ones(10) / 10, mode='valid')
                axes[1, 0].plot(range(9, len(self.episode_constraint_violations)), ma, 'b-', linewidth=2, label='MA(10)')
            axes[1, 0].set_title('每轮PSH约束违反次数', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('训练轮次')
            axes[1, 0].set_ylabel('违反次数')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Actor损失
            if len(self.agent.episode_actor_losses) > 0:
                # 过滤NaN
                actor_losses = np.array(self.agent.episode_actor_losses)
                valid_indices = ~np.isnan(actor_losses)
                if valid_indices.any():
                    axes[1, 1].plot(np.where(valid_indices)[0], actor_losses[valid_indices], 
                                   alpha=0.6, color='blue', label='Actor Loss')
                    # 计算移动平均
                    valid_losses = actor_losses[valid_indices]
                    if len(valid_losses) >= 10:
                        ma = np.convolve(valid_losses, np.ones(10) / 10, mode='valid')
                        axes[1, 1].plot(range(9, len(valid_losses)), ma, 'r-', linewidth=2, label='MA(10)')
                axes[1, 1].set_title('Actor损失', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('训练轮次')
                axes[1, 1].set_ylabel('损失值')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Critic损失
            if len(self.agent.episode_critic_losses) > 0:
                # 过滤NaN
                critic_losses = np.array(self.agent.episode_critic_losses)
                valid_indices = ~np.isnan(critic_losses)
                if valid_indices.any():
                    axes[1, 2].plot(np.where(valid_indices)[0], critic_losses[valid_indices], 
                                   alpha=0.6, color='green', label='Critic Loss')
                    # 计算移动平均
                    valid_losses = critic_losses[valid_indices]
                    if len(valid_losses) >= 10:
                        ma = np.convolve(valid_losses, np.ones(10) / 10, mode='valid')
                        axes[1, 2].plot(range(9, len(valid_losses)), ma, 'r-', linewidth=2, label='MA(10)')
                axes[1, 2].set_title('Critic损失', fontsize=12, fontweight='bold')
                axes[1, 2].set_xlabel('训练轮次')
                axes[1, 2].set_ylabel('损失值')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            # 7. 总损失
            if len(self.agent.episode_total_losses) > 0:
                # 过滤NaN
                total_losses = np.array(self.agent.episode_total_losses)
                valid_indices = ~np.isnan(total_losses)
                if valid_indices.any():
                    axes[2, 0].plot(np.where(valid_indices)[0], total_losses[valid_indices], 
                                   alpha=0.6, color='purple', label='Total Loss')
                    valid_losses = total_losses[valid_indices]
                    if len(valid_losses) >= 10:
                        ma = np.convolve(valid_losses, np.ones(10) / 10, mode='valid')
                        axes[2, 0].plot(range(9, len(valid_losses)), ma, 'r-', linewidth=2, label='MA(10)')
                axes[2, 0].set_title('总损失', fontsize=12, fontweight='bold')
                axes[2, 0].set_xlabel('训练轮次')
                axes[2, 0].set_ylabel('损失值')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
            
            # 8. 熵损失
            if len(self.agent.episode_entropy_losses) > 0:
                # 过滤NaN
                entropy_losses = np.array(self.agent.episode_entropy_losses)
                valid_indices = ~np.isnan(entropy_losses)
                if valid_indices.any():
                    axes[2, 1].plot(np.where(valid_indices)[0], entropy_losses[valid_indices], 
                                   alpha=0.6, color='brown', label='Entropy Loss')
                    valid_losses = entropy_losses[valid_indices]
                    if len(valid_losses) >= 10:
                        ma = np.convolve(valid_losses, np.ones(10) / 10, mode='valid')
                        axes[2, 1].plot(range(9, len(valid_losses)), ma, 'r-', linewidth=2, label='MA(10)')
                axes[2, 1].set_title('熵损失', fontsize=12, fontweight='bold')
                axes[2, 1].set_xlabel('训练轮次')
                axes[2, 1].set_ylabel('损失值')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
            
            # 9. 奖励分布直方图
            if len(self.episode_rewards) > 0:
                valid_rewards = [r for r in self.episode_rewards if not np.isnan(r) and not np.isinf(r)]
                if valid_rewards:
                    axes[2, 2].hist(valid_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[2, 2].axvline(np.mean(valid_rewards), color='r', linestyle='--', 
                                      linewidth=2, label=f'均值: {np.mean(valid_rewards):.2f}')
                    axes[2, 2].set_title('奖励分布', fontsize=12, fontweight='bold')
                    axes[2, 2].set_xlabel('平均奖励')
                    axes[2, 2].set_ylabel('频次')
                    axes[2, 2].legend()
                    axes[2, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plot_save_path, dpi=150, bbox_inches='tight')
            print(f"训练可视化已保存到: {self.plot_save_path}")
            plt.close()
        except Exception as e:
            print(f"绘制可视化出错: {e}")
            import traceback
            traceback.print_exc()
