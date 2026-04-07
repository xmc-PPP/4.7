"""
Deep Deterministic Policy Gradient (DDPG) 算法实现 - 方案2: 混合动作空间

修改说明:
- 支持混合动作空间 (PSH离散 + BESS连续)
- Actor输出3维: [psh_action, bess1_action, bess2_action]
- PSH动作离散化为 {0:保持, 1:启动发电, 2:启动抽水, 3:停止}
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


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
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
    """
    Actor网络 - 输出混合动作空间
    
    输出3维动作:
    - action[0]: PSH (连续值[-1,1]映射到离散动作)
    - action[1]: BESS1 (连续值[-1, 1])
    - action[2]: BESS2 (连续值[-1, 1])
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int = 3,
            hidden_dims: List[int] = [256, 256],
            init_w: float = 3e-3
    ):
        super(Actor, self).__init__()

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, action_dim)

        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer, ln in zip(self.layers, self.layer_norms):
            x = F.relu(ln(layer(x)))
        action = torch.tanh(self.output_layer(x))
        return action


class Critic(nn.Module):
    """Critic网络"""

    def __init__(
            self,
            state_dim: int,
            action_dim: int = 3,
            hidden_dims: List[int] = [256, 256],
            init_w: float = 3e-3
    ):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        prev_dim = hidden_dims[0] + action_dim
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0)

        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = torch.cat([x, action], dim=1)

        for layer, ln in zip(self.layers, self.layer_norms):
            x = F.relu(ln(layer(x)))

        q_value = self.output_layer(x)
        return q_value


class DDPGAgent:
    """
    DDPG智能体 - 混合动作空间版本
    
    动作空间:
    - action[0]: PSH (连续值[-1,1]映射到离散{0,1,2,3})
    - action[1]: BESS1 (连续值[-1,1])
    - action[2]: BESS2 (连续值[-1,1])
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int = 3,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            gamma: float = 0.995,
            tau: float = 0.001,
            buffer_capacity: int = 100000,
            batch_size: int = 256,
            hidden_dims: List[int] = [256, 256],
            warmup_steps: int = 5000,
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

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 噪声参数
        self.noise_std = 0.2
        self.noise_decay = 0.9995
        self.min_noise_std = 0.05

        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        self.q_values_target = []

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作 - 返回混合动作空间"""
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

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        rewards = torch.clamp(rewards, -10, 10)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            target_q = torch.clamp(target_q, -100, 100)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0)
        if self.total_steps % 2 == 0:
            pred_actions = self.actor(states)
            actor_loss = -self.critic(states, pred_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.total_steps += 1

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
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
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

        if 'actor_scheduler' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])

        self.noise_std = checkpoint.get('noise_std', 0.1)
        self.total_steps = checkpoint.get('total_steps', 0)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


class DDPGTrainer:
    """DDPG训练器 - 混合动作空间版本"""

    def __init__(
            self,
            env,
            agent: DDPGAgent,
            max_episodes: int = 1000,
            max_steps_per_episode: int = 96,
            eval_interval: int = 50,
            save_interval: int = 100,
            log_interval: int = 10
    ):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval

        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.episode_voltage_violations = []
        self.episode_soc_stats = []
        self.episode_power_stats = []
        self.psh_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    def train(self):
        """训练智能体"""
        print("开始训练...")
        print(f"预热期: 前{self.agent.warmup_steps}步只收集经验不更新网络")
        print("动作空间: PSH离散[0-3] + BESS1连续[-1,1] + BESS2连续[-1,1]")

        best_eval_reward = -np.inf
        patience = 0

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            episode_violations = 0
            episode_convergence_failures = 0
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

                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                update_info = self.agent.update()

                episode_reward += reward
                episode_steps += 1

                if not info.get('converged', True):
                    episode_convergence_failures += 1

                episode_violations += info.get('v_violation_count', 0)

                if 'psh_action' in info:
                    episode_psh_actions[info['psh_action']] += 1

                if 'voltage' in info:
                    voltages_min.append(np.min(info['voltage']))
                    voltages_max.append(np.max(info['voltage']))

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

            self.agent.decay_noise()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.episode_voltage_violations.append(episode_violations)

            for k, v in episode_psh_actions.items():
                self.psh_action_counts[k] += v

            soc_stats = {
                'psh_upper_mean': np.mean(psh_upper_socs),
                'psh_upper_min': np.min(psh_upper_socs),
                'psh_upper_max': np.max(psh_upper_socs),
                'psh_lower_mean': np.mean(psh_lower_socs),
                'psh_lower_min': np.min(psh_lower_socs),
                'psh_lower_max': np.max(psh_lower_socs),
                'bess1_mean': np.mean(bess1_socs),
                'bess2_mean': np.mean(bess2_socs)
            }
            self.episode_soc_stats.append(soc_stats)

            power_stats = {
                'psh_mean': np.mean(np.abs(psh_powers)),
                'bess1_mean': np.mean(np.abs(bess1_powers)),
                'bess2_mean': np.mean(np.abs(bess2_powers))
            }
            self.episode_power_stats.append(power_stats)

            if (episode + 1) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_violations = np.mean(self.episode_voltage_violations[-self.log_interval:])

                print(f"\nEpisode {episode + 1}/{self.max_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f} | Noise Std: {self.agent.noise_std:.4f}")
                print(f"  Avg Voltage Violations: {avg_violations:.2f} per episode")
                print(f"  PSH Actions: Hold={episode_psh_actions[0]}, Gen={episode_psh_actions[1]}, Pump={episode_psh_actions[2]}, Stop={episode_psh_actions[3]}")
                print(f"  PSH Upper SOC: {soc_stats['psh_upper_mean']:.3f}")
                print(f"  PSH Lower SOC: {soc_stats['psh_lower_mean']:.3f}")

                if len(voltages_min) > 0:
                    print(f"  Voltage Range: [{np.min(voltages_min):.4f}, {np.max(voltages_max):.4f}] p.u.")

            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                print(f"\n>>> Evaluation at Episode {episode + 1}: {eval_reward:.2f}")

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save("best_model.pth")
                    patience = 0
                else:
                    patience += 1

                if patience > 10:
                    for param_group in self.agent.actor_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    for param_group in self.agent.critic_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print("学习率降低50%")
                    patience = 0

            if (episode + 1) % self.save_interval == 0:
                self.agent.save(f"ddpg_checkpoint_episode_{episode + 1}.pth")

        print("训练完成!")
        print(f"\nPSH动作统计 (总计):")
        print(f"  Hold(保持): {self.psh_action_counts[0]}")
        print(f"  StartGen(启动发电): {self.psh_action_counts[1]}")
        print(f"  StartPump(启动抽水): {self.psh_action_counts[2]}")
        print(f"  Stop(停止): {self.psh_action_counts[3]}")

    def evaluate(self, num_episodes: int = 5) -> float:
        """评估智能体"""
        total_reward = 0
        total_violations = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_violations = 0

            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state, add_noise=False)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward

                if 'v_violations' in info:
                    episode_violations += len(info['v_violations'])

                state = next_state

                if done:
                    break

            total_reward += episode_reward
            total_violations += episode_violations

        avg_reward = total_reward / num_episodes
        avg_violations = total_violations / num_episodes
        print(f"  Eval Avg Violations: {avg_violations:.2f}")

        return avg_reward

    def plot_training_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Raw')
        if len(self.episode_rewards) >= 100:
            ma = np.convolve(self.episode_rewards, np.ones(100) / 100, mode='valid')
            axes[0, 0].plot(range(99, len(self.episode_rewards)), ma, 'r-', linewidth=2, label='MA(100)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if len(self.eval_rewards) > 0:
            eval_episodes = range(self.eval_interval, len(self.eval_rewards) * self.eval_interval + 1,
                                  self.eval_interval)
            axes[0, 1].plot(eval_episodes, self.eval_rewards, 'bo-', markersize=4)
            axes[0, 1].set_title('Evaluation Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(self.episode_voltage_violations, alpha=0.6)
        if len(self.episode_voltage_violations) >= 100:
            ma_viol = np.convolve(self.episode_voltage_violations, np.ones(100) / 100, mode='valid')
            axes[0, 2].plot(range(99, len(self.episode_voltage_violations)), ma_viol, 'r-', linewidth=2)
        axes[0, 2].set_title('Voltage Violations per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Violation Count')
        axes[0, 2].grid(True, alpha=0.3)

        if len(self.agent.actor_losses) > 0:
            axes[1, 0].plot(self.agent.actor_losses, alpha=0.6)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)

        if len(self.agent.critic_losses) > 0:
            axes[1, 1].plot(self.agent.critic_losses, alpha=0.6)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_ylim(0, min(100, np.percentile(self.agent.critic_losses, 95) * 2))
            axes[1, 1].grid(True, alpha=0.3)

        if len(self.agent.q_values) > 0:
            axes[1, 2].plot(self.agent.q_values, alpha=0.6, label='Current Q')
            if len(self.agent.q_values_target) > 0:
                axes[1, 2].plot(self.agent.q_values_target, alpha=0.6, label='Target Q')
            axes[1, 2].set_title('Q Values')
            axes[1, 2].set_xlabel('Update Step')
            axes[1, 2].set_ylabel('Q Value')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
