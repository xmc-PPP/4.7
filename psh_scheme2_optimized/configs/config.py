"""
配置文件 - 方案2: 统一DRL框架，混合动作空间
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PSHConfig:
    """抽水储能配置 - 定速运行，上下水库，离散动作控制"""
    unit_id: int = 1
    node_id: int = 34
    
    rated_generation_power: float = 5.0
    rated_pumping_power: float = 5.0
    
    upper_reservoir_capacity: float = 30.0
    lower_reservoir_capacity: float = 30.0
    upper_reservoir_min: float = 3.0
    lower_reservoir_min: float = 3.0
    
    generation_efficiency: float = 0.85
    pumping_efficiency: float = 0.85
    
    initial_upper_soc: float = 0.5
    initial_lower_soc: float = 0.5
    
    max_daily_cycles: int = 4
    min_operation_duration: int = 8
    max_operation_duration: int = 24


@dataclass
class BESSConfig:
    """电池储能配置"""
    unit_id: int = 2
    node_id: int = 16
    max_power: float = 2.0
    capacity: float = 4.0
    min_soc: float = 0.2
    max_soc: float = 0.9
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc: float = 0.5
    ramp_rate_limit: float = 1.0


@dataclass
class EnvironmentConfig:
    """环境配置"""
    node_file: str = "data/node_data.csv"
    line_file: str = "data/line_data.csv"
    load_file: str = "data/load_data.csv"
    pv_file: str = "data/pv_data.csv"
    price_file: str = "data/price_data.csv"
    
    time_step: float = 0.25
    episode_length: int = 96

    psh_config: PSHConfig = field(default_factory=PSHConfig)
    bess_configs: List[BESSConfig] = field(default_factory=lambda: [
        BESSConfig(unit_id=2, node_id=16),
        BESSConfig(unit_id=3, node_id=27)
    ])


@dataclass
class DDPGConfig:
    """DDPG算法配置 - 混合动作空间"""
    state_dim: int = 46
    action_dim: int = 3
    
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    actor_lr: float = 1e-4
    critic_lr: float = 1e-4

    gamma: float = 0.995
    tau: float = 0.001
    buffer_capacity: int = 100000
    batch_size: int = 256

    initial_noise_std: float = 0.2
    noise_decay: float = 0.9995
    min_noise_std: float = 0.05

    warmup_steps: int = 5000

    device: str = 'cpu'


@dataclass
class TrainingConfig:
    """训练配置"""
    max_episodes: int = 1000
    max_steps_per_episode: int = 96
    eval_interval: int = 50
    save_interval: int = 100
    log_interval: int = 10

    random_seed: int = 42

    model_save_path: str = "models/ddpg_model.pth"
    checkpoint_dir: str = "checkpoints/"


@dataclass
class EvaluationConfig:
    """评估配置"""
    num_eval_episodes: int = 10
    render: bool = True
    save_results: bool = True
    results_dir: str = "results/"


# 全局配置实例
psh_config = PSHConfig()
bess_configs = [
    BESSConfig(unit_id=2, node_id=16),
    BESSConfig(unit_id=3, node_id=27)
]
env_config = EnvironmentConfig()
ddpg_config = DDPGConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig()
