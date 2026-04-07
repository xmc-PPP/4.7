# 抽水储能与电池储能协同调度 - 方案2: 统一DRL框架

## 项目概述

本项目实现了**抽水储能(PSH)**与**电池储能(BESS)**在配电网中的协同优化调度，采用**统一DRL框架**（方案2）。

## 方案2特点

```
┌─────────────────────────────────────────────────────────────────┐
│                     统一DRL智能体 (DDPG)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  状态空间 (46维):                                          │  │
│  │    - 电网状态: 负荷、光伏、电价 (34节点)                    │  │
│  │    - PSH状态: 上下库SOC、功率、模式、运行时长、启停次数      │  │
│  │    - BESS状态: SOC、功率 (2个BESS)                         │  │
│  │    - 时间特征                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  混合动作空间 (3维):                                        │  │
│  │    - action[0]: PSH离散 {0:保持, 1:启动发电, 2:启动抽水, 3:停止}│  │
│  │    - action[1]: BESS1连续 [-1, 1]                          │  │
│  │    - action[2]: BESS2连续 [-1, 1]                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
├─────────────────────────────────────────────────────────────────┤
│                    IEEE 34节点配电网                             │
│              节点34: PSH (定速功率)                               │
│              节点16: BESS1 (连续调节)                             │
│              节点27: BESS2 (连续调节)                             │
└─────────────────────────────────────────────────────────────────┘
```

## 关键特性

### 1. 混合动作空间

| 动作维度 | 类型 | 含义 | 范围 |
|----------|------|------|------|
| action[0] | 离散 | PSH控制 | {0:保持, 1:启动发电, 2:启动抽水, 3:停止} |
| action[1] | 连续 | BESS1功率 | [-1, 1] |
| action[2] | 连续 | BESS2功率 | [-1, 1] |

**离散化映射规则:**
```python
if action < -0.5:    -> 3 (STOP)
elif action < 0.0:   -> 0 (HOLD)
elif action < 0.5:   -> 1 (START_GENERATE)
else:                -> 2 (START_PUMP)
```

### 2. PSH运行约束

- **定速运行**: 发电/抽水功率固定 (5MW)
- **上下水库**: 分别建模，考虑水力联系
- **启停限制**: 每日最多4次
- **运行时长**: 每次2-6小时

### 3. 储能节点配置

| 储能 | 节点 | 功率 | 容量 |
|------|------|------|------|
| PSH | 34 | 5MW (定速) | 30MWh (上下库各30MWh) |
| BESS1 | 16 | 2MW | 4MWh |
| BESS2 | 27 | 2MW | 4MWh |

## 安装依赖

```bash
pip install numpy pandas torch matplotlib scipy gymnasium
```

## 使用方法

### 1. 完整流程

```bash
cd /mnt/okcomputer/output/psh_scheme2
python main.py --mode all
```

### 2. 分步执行

```bash
# 仅生成数据
python main.py --mode generate_data

# 仅训练
python main.py --mode train

# 仅评估
python main.py --mode eval --model_path models/ddpg_model.pth
```

### 3. 自定义配置

修改 `configs/config.py` 中的参数。

## 项目结构

```
psh_scheme2/
├── main.py                      # 主程序入口
├── README.md                    # 本文件
├── models/
│   └── pumped_storage.py        # PSH离散动作 + BESS
├── envs/
│   └── distribution_network.py  # 混合动作空间环境
├── algorithms/
│   └── ddpg.py                  # DDPG (混合动作空间)
├── utils/
│   └── data_generator.py        # 数据生成
└── configs/
    └── config.py                # 配置参数
```

## 核心类说明

### PumpedStorageUnit (定速抽水储能)

```python
class PumpedStorageUnit:
    """
    定速抽水储能机组 - 离散动作控制
    
    离散动作:
    - 0 (HOLD): 保持当前状态
    - 1 (START_GENERATE): 启动发电
    - 2 (START_PUMP): 启动抽水
    - 3 (STOP): 停止运行
    """
    
    def step(action: int) -> (power, info)
        """执行离散动作"""
```

### DistributionNetworkEnv (混合动作空间环境)

```python
class DistributionNetworkEnv:
    """
    配电网环境 - 混合动作空间
    
    动作空间: Box(low=-1, high=1, shape=(3,))
        - action[0]: PSH (连续值映射到离散动作)
        - action[1]: BESS1 (连续值)
        - action[2]: BESS2 (连续值)
    """
    
    def _discretize_psh_action(continuous_action: float) -> int:
        """将连续动作离散化"""
```

## 算法流程

```
对于每个训练回合:
    1. 重置环境 (PSH和BESS状态初始化)
    
    2. 对于每个时间步 (0-95, 15分钟分辨率):
       a. DRL选择动作: [psh_action, bess1_action, bess2_action]
       
       b. 解析动作:
          - psh_action (连续[-1,1]) -> 离散{0,1,2,3}
          - bess1_action, bess2_action (连续[-1,1]) 直接使用
       
       c. 执行动作:
          - PSH执行离散动作 (定速运行)
          - BESS执行连续动作
       
       d. 潮流计算
       
       e. 计算奖励
       
       f. 存储经验 (连续动作用于训练)
       
       g. 更新DRL网络
```

## 输出结果

训练完成后，结果保存在 `results/` 目录:

- `training_stats.csv`: 训练统计
- `evaluation_data.csv`: 评估数据
- `training_history.png`: 训练历史

## 参考文献

1. Su Y, et al. Two-stage optimal dispatch framework of active distribution networks with 
   hybrid energy storage systems via deep reinforcement learning and real-time feedback dispatch.
   Journal of Energy Storage, 2025.

2. 小型抽水蓄能参数资料与应用场景汇编

3. Kersting W H. Radial distribution test feeders. IEEE Transactions on Power Systems, 1991.
