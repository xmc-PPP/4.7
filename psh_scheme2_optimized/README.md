# 抽水储能与电池储能协同调度 - 优化版

本项目是对原方案2的优化版本，修复了三个主要问题：上下水库关联逻辑、潮流计算失效、奖励函数归一化。

## 项目特点

- **统一DRL框架**: 单一DDPG智能体统一决策PSH和BESS
- **混合动作空间**: PSH离散动作 + BESS连续动作
- **IEEE 34节点配电网**: 包含PSH和BESS的协同优化

## 储能配置

- **PSH**: 节点34，定速5MW运行，上下水库各30MWh
- **BESS1**: 节点16，2MW/4MWh
- **BESS2**: 节点27，2MW/4MWh

## 修复内容

1. **上下水库关联逻辑**: 发电时上库水减少、下库增加；抽水时相反
2. **潮流计算**: 使用pandapower确保准确性和收敛性
3. **奖励函数归一化**: 各分量归一化到相近数量级

## 安装依赖

```bash
pip install gym numpy pandas matplotlib torch pandapower
```

## 运行方式

```bash
# 完整流程（生成数据、训练、评估）
python main.py --mode all

# 仅生成数据
python main.py --mode generate_data

# 仅训练
python main.py --mode train

# 仅评估
python main.py --mode eval --model_path models/ddpg_model.pth
```

## 项目结构

```
psh_scheme2/
├── algorithms/          # DDPG算法实现
├── configs/            # 配置文件
├── data/               # 数据文件
├── envs/               # 环境实现
├── models/             # 储能模型
├── utils/              # 工具函数
├── main.py             # 主运行脚本
├── MODIFICATIONS.md    # 修改说明文档
└── README.md           # 本文件
```

## 详细修改说明

参见 [MODIFICATIONS.md](MODIFICATIONS.md) 文件。

## 原项目

https://github.com/xmc-PPP/scheme2
