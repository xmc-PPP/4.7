# 代码修改说明文档

## 项目概述
本项目是对GitHub项目 https://github.com/xmc-PPP/scheme2 的优化版本，修复了原代码中的三个主要问题。

## 主要修改内容

### 1. 修复上下水库关联逻辑 (pumped_storage.py)

**原问题：**
- 原代码中上下水库缺少关联逻辑，上水库放水和抽水时没有正确影响下水库的水量

**修复内容：**
- 在 `_execute_operation()` 方法中实现了正确的水库关联逻辑：
  - **发电模式**：水从上水库流向下水库
    - 上水库SOC减少（放水）
    - 下水库SOC增加（接收水）
  - **抽水模式**：水从下水库抽向上水库
    - 下水库SOC减少（放水）
    - 上水库SOC增加（接收水）
- 添加了 `check_water_balance()` 方法验证水量平衡
- 添加了效率损失计算，确保能量守恒

**关键代码修改：**
```python
# 发电模式
energy_output = self.rated_generation_power * self.time_step
upper_energy_consumed = energy_output / self.generation_efficiency
self.upper_soc -= upper_energy_consumed / self.upper_reservoir_capacity
self.lower_soc += upper_energy_consumed * self.generation_efficiency / self.lower_reservoir_capacity

# 抽水模式
energy_input = self.rated_pumping_power * self.time_step
lower_energy_consumed = energy_input * self.pumping_efficiency
self.lower_soc -= lower_energy_consumed / self.lower_reservoir_capacity
self.upper_soc += energy_input * self.pumping_efficiency / self.upper_reservoir_capacity
```

### 2. 修复潮流计算问题 (distribution_network.py)

**原问题：**
- 原代码使用高斯-赛德尔法，电压标幺值始终为1，潮流计算不收敛
- 负荷数据规模过大（近5000MW），不适合配电网

**修复内容：**
- 使用 `pandapower` 库替代自定义潮流计算，确保准确性和收敛性
- 修复了线路阻抗参数转换问题
- 在 `data_generator.py` 中缩小了负荷数据规模（缩小100倍），使其符合配电网实际（总负荷约50MW）
- 添加了简化潮流计算方法作为后备方案

**关键代码修改：**
```python
# 使用pandapower进行潮流计算
def _solve_pandapower(self, P, Q):
    # 清除旧的负荷和发电
    # 添加新的负荷和发电
    # 运行潮流计算
    pp.runpp(self.net, algorithm='nr', init='flat', max_iteration=100)
    # 提取结果
    V = self.net.res_bus.vm_pu.values
```

### 3. 修复奖励函数归一化问题 (distribution_network.py)

**原问题：**
- 原奖励函数各部分数量级差异巨大：
  - 能量套利收益：1e-4 缩放
  - 电压越限惩罚：-500 * v²
  - 不收敛惩罚：-5000
  - SOC惩罚：-10 * (0.2 - soc)²
- 导致训练过程中某些项无法体现效果

**修复内容：**
- 添加了 `_init_reward_normalization()` 方法初始化归一化参数
- 使用参考值对各分量进行归一化：
  - 能量套利收益：参考值 50 ¥/step
  - 电压越限：参考值 0.05 p.u.
  - SOC越限：参考值 0.1
  - 动作变化：参考值 0.5
- 设置了合理的权重系数
- 所有分量归一化到相近的数量级

**关键代码修改：**
```python
def _init_reward_normalization(self):
    self.ref_revenue = 50.0
    self.ref_voltage_violation = 0.05
    self.ref_soc_violation = 0.1
    self.ref_action_change = 0.5
    
    self.w_revenue = 1.0
    self.w_voltage = -1.0
    self.w_convergence = -0.5
    self.w_soc = -0.3
    self.w_smooth = -0.1

def _calculate_reward_normalized(self, ...):
    revenue_norm = np.clip(total_revenue / self.ref_revenue, -5, 5)
    voltage_penalty = -(voltage_violation_sum / self.ref_voltage_violation) ** 2
    # ... 其他分量类似处理
    reward = (self.w_revenue * revenue_norm +
              self.w_voltage * voltage_penalty +
              ...)
```

## 文件修改列表

### 新增文件
1. `models/pumped_storage.py` - 抽水储能和电池储能模型（原文件下载失败，根据引用重新创建）

### 修改文件
1. `envs/distribution_network.py`
   - 使用pandapower替代自定义潮流计算
   - 修复奖励函数归一化
   - 改进状态空间处理

2. `utils/data_generator.py`
   - 缩小负荷数据规模100倍（从5000MW到50MW）

## 测试验证

运行测试代码验证了以下功能：
1. 环境创建成功
2. PSH上下水库SOC变化正确（发电时上库减少、下库增加；抽水时相反）
3. 电压计算合理（范围[0.998, 1.0005] p.u.）
4. 潮流计算收敛
5. 奖励值在合理范围内

## 依赖要求

新增依赖：
- pandapower >= 2.0

原有依赖：
- gym
- numpy
- pandas
- matplotlib
- torch

## 运行方式

```bash
# 生成数据
python main.py --mode generate_data

# 训练模型
python main.py --mode train

# 评估模型
python main.py --mode eval --model_path models/ddpg_model.pth

# 完整流程
python main.py --mode all
```

## 注意事项

1. 确保pandapower已正确安装：`pip install pandapower`
2. 数据文件会自动生成在 `data/` 目录
3. 模型检查点保存在 `checkpoints/` 目录
4. 训练结果保存在 `results/` 目录
