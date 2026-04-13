# DDPG与PPO算法对比分析及奖励函数详解

## 抽水储能调度系统 - 版本4.9.2

---

## 第一部分：DDPG与PPO在本项目中的优缺点分析

### 一、动作空间适配性对比

#### PPO（本项目采用）

**动作空间设计**：
```python
# PPO输出离散动作
self.action_space = Discrete(4)  # 保持、发电、抽水、停止
```

**优点**：
1. **天然适配离散动作**：PSH调度问题本质是离散决策（4种运行模式），PPO直接输出动作概率分布
2. **策略解释性强**：每个动作有明确的概率，便于调试和分析
3. **无需额外转换**：避免连续到离散的映射误差

**缺点**：
1. **动作粒度受限**：只能输出预定义的离散动作，无法精细控制功率大小
2. **维度灾难风险**：如果动作空间增大，训练难度显著增加

---

#### DDPG

**动作空间设计**：
```python
# DDPG输出连续动作 [-1, 1]
action = torch.tanh(self.output_layer(x))  # 输出范围[-1, 1]

# 需要映射到离散动作
def _discretize_psh_action(self, continuous_action: float) -> int:
    if continuous_action < -0.6:
        return PSHAction.STOP
    elif continuous_action < -0.2:
        return PSHAction.HOLD
    elif continuous_action < 0.2:
        return PSHAction.START_GENERATE
    elif continuous_action < 0.6:
        return PSHAction.START_PUMP
    else:
        return PSHAction.HOLD
```

**优点**：
1. **理论上可精细控制**：连续输出理论上可以实现任意功率值
2. **平滑策略**：连续动作空间策略更平滑，有利于稳定性

**缺点**：
1. **离散化误差**：连续到离散的映射会丢失信息
2. **边界问题**：映射阈值(-0.6, -0.2, 0.2, 0.6)需要精心调整
3. **探索困难**：在离散动作边界处探索效率低

**本项目结论**：PSH调度是离散决策问题，PPO更适合。

---

### 二、训练稳定性对比

#### PPO

**核心机制 - 裁剪目标**：
```python
# PPO裁剪机制
ratio = torch.exp(log_probs - batch_old_log_probs)
surr1 = ratio * batch_advantages
surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
actor_loss = -torch.min(surr1, surr2).mean()
```

**稳定性优势**：
1. **策略更新受限**：裁剪机制限制策略变化幅度，防止策略突变
2. **单调性保证**：理论保证策略性能不会大幅下降
3. **样本复用**：同一批数据可以多次更新，提高稳定性

**实际表现**：
- 200轮训练无NaN问题
- 奖励曲线平滑上升
- PSH约束达标率100%

---

#### DDPG

**核心机制 - 软更新**：
```python
# DDPG软更新
def _soft_update(self, source: nn.Module, target: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - self.tau) + param.data * self.tau
        )
```

**稳定性问题**：
1. **目标网络延迟**：软更新导致目标Q值滞后，影响训练稳定性
2. **Q值过估计**：Critic容易过估计Q值，导致策略退化
3. **超参数敏感**：τ值需要精心调整，过大不稳定，过小收敛慢

**实际表现（历史版本）**：
- 需要大量调参才能稳定
- 容易出现Q值爆炸
- 收敛速度较慢

**本项目结论**：PPO的裁剪机制更适合需要严格约束的电力系统调度问题。

---

### 三、样本效率对比

#### PPO

**样本使用方式**：
```python
# PPO多轮更新
for epoch in range(n_epochs):  # 通常4-10轮
    for batch in data:
        update(batch)  # 同一样本多次使用
```

**效率优势**：
1. **样本复用**：同一批样本可以更新4-10次
2. **在线学习**：不需要大容量回放缓冲区
3. **即时反馈**：每轮收集的数据立即用于更新

**计算成本**：
- 内存占用：低（无需回放缓冲区）
- 计算量：中等（多轮更新但数据量小）

---

#### DDPG

**样本使用方式**：
```python
# DDPG经验回放
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)  # 通常100000

# 每次采样更新
batch = replay_buffer.sample(batch_size)  # 每个样本只用一次
update(batch)
```

**效率问题**：
1. **样本浪费**：每个样本通常只用一次
2. **内存需求**：需要大容量回放缓冲区
3. **冷启动问题**：前期缓冲区未满时无法更新

**计算成本**：
- 内存占用：高（100K条经验 ≈ 几百MB）
- 计算量：中等（单次更新但数据量大）

**本项目结论**：电力系统仿真成本高，PPO的样本复用更有优势。

---

### 四、探索机制对比

#### PPO

**探索方式 - 随机策略采样**：
```python
# PPO探索机制
probs = F.softmax(action_logits, dim=-1)
dist = Categorical(probs)
action = dist.sample()  # 按概率随机采样

# 熵正则化
entropy_loss = -entropy.mean()  # 鼓励高熵（更多探索）
```

**探索优势**：
1. **内在探索**：探索是策略的内在属性，无需额外设计
2. **自动衰减**：随着训练进行，熵自动减小，策略趋于确定
3. **理论保证**：熵正则化有理论收敛保证

---

#### DDPG

**探索方式 - 外部噪声**：
```python
# DDPG探索机制
if add_noise:
    noise = np.random.normal(0, self.noise_std, size=self.action_dim)
    action = action + noise

# 噪声衰减
def decay_noise(self):
    self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)
```

**探索问题**：
1. **需要手动设计**：噪声类型、初始值、衰减率都需要调整
2. **探索不均**：在连续空间均匀探索效率低
3. **噪声敏感**：噪声过大破坏策略，过小探索不足

**本项目结论**：PPO的内在探索机制更优雅，更适合本项目。

---

### 五、网络架构复杂度对比

#### PPO

**网络结构**：
```
Actor:  state -> [256] -> [256] -> action_probs
Critic: state -> [256] -> [256] -> value
```

**复杂度**：
- 网络数量：2个
- 参数总量：约 2 × (state_dim × 256 + 256 × 256 + 256 × action_dim) ≈ 300K

---

#### DDPG

**网络结构**：
```
Actor:        state -> [256] -> [256] -> action
Actor_target: state -> [256] -> [256] -> action (副本)
Critic:       state + action -> [256] -> [256] -> Q_value
Critic_target: state + action -> [256] -> [256] -> Q_value (副本)
```

**复杂度**：
- 网络数量：4个
- 参数总量：约 4 × 300K = 1.2M

**本项目结论**：PPO架构更简洁，参数更少，更容易训练。

---

### 六、在本项目中的综合对比

| 维度 | PPO | DDPG | 本项目选择 |
|------|-----|------|-----------|
| 动作空间适配 | ✅ 离散动作天然支持 | ❌ 需要离散化映射 | PPO |
| 训练稳定性 | ✅ 裁剪机制保证 | ⚠️ 需要精心调参 | PPO |
| 样本效率 | ✅ 可复用4-10次 | ❌ 每个样本用1次 | PPO |
| 探索机制 | ✅ 内在熵正则化 | ❌ 外部噪声 | PPO |
| 架构复杂度 | ✅ 2个网络 | ❌ 4个网络 | PPO |
| 收敛速度 | ✅ 快 | ⚠️ 慢 | PPO |
| 超参数敏感 | ✅ 低 | ❌ 高 | PPO |

**最终结论**：对于本项目的离散动作、高约束、高仿真成本场景，PPO是更优选择。

---

## 第二部分：奖励函数详细解析

### 一、奖励函数完整表达式

```python
def _calculate_reward(
    self,
    price: float,                    # 当前电价
    psh_power: float,                # PSH功率（正为发电，负为抽水）
    bess_powers: List[float],        # BESS功率列表
    v_violation_mag: np.ndarray,     # 电压越限幅度
    converged: bool,                 # 潮流是否收敛
    psh_info: Dict,                  # PSH状态信息
    bess_socs: List[float]           # BESS SOC列表
) -> float:
```

### 二、奖励函数各组成部分详解

#### 1. 能量套利收益 (Revenue)

```python
# 计算收益
psh_revenue = price * psh_power * self.time_step
bess_revenue = sum(price * power * self.time_step for power in bess_powers)
total_revenue = psh_revenue + bess_revenue

# 归一化
revenue_norm = total_revenue / self.ref_revenue  # ref_revenue = 5.0
```

**物理意义**：
- 当PSH发电时（psh_power > 0），向电网供电获得收益
- 当PSH抽水时（psh_power < 0），从电网购电产生成本
- 鼓励在电价高时发电、电价低时抽水

**权重**：`w_revenue = 1.0`

---

#### 2. 电压越限惩罚 (Voltage Penalty)

```python
# 计算电压越限幅度
voltage_violation_sum = np.sum(v_violation_mag)

# 二次惩罚
voltage_penalty = -(voltage_violation_sum / self.ref_voltage_violation) ** 2
# ref_voltage_violation = 0.02
```

**物理意义**：
- 惩罚电压超出安全范围（0.95-1.05 p.u.）的行为
- 二次惩罚：越限越多惩罚越重
- 保证电网电压稳定

**权重**：`w_voltage = -5.0`

---

#### 3. SOC约束惩罚 (SOC Penalty)

```python
soc_penalty = 0.0

# PSH上水库SOC边界惩罚
if psh_upper_soc > 0.80:
    soc_penalty -= ((psh_upper_soc - 0.80) / self.ref_soc_violation) ** 2
elif psh_upper_soc < 0.20:
    soc_penalty -= ((0.20 - psh_upper_soc) / self.ref_soc_violation) ** 2

# PSH下水库SOC边界惩罚
if psh_lower_soc > 0.80:
    soc_penalty -= ((psh_lower_soc - 0.80) / self.ref_soc_violation) ** 2
elif psh_lower_soc < 0.20:
    soc_penalty -= ((0.20 - psh_lower_soc) / self.ref_soc_violation) ** 2
```

**物理意义**：
- 惩罚SOC接近边界（<0.2或>0.8）的行为
- 保护水库不过度充放电
- 预留安全裕度应对突发情况

**权重**：`w_soc = -3.0`

---

#### 4. SOC平衡奖励 (Balance Reward)

```python
# 鼓励上下水库SOC保持平衡
soc_balance = 1.0 - abs(psh_upper_soc - psh_lower_soc)
balance_reward = self.w_balance * soc_balance  # w_balance = 0.3
```

**物理意义**：
- 鼓励上下水库能量保持平衡
- 避免单一水库过度使用
- 提高系统整体可用容量

---

#### 5. PSH约束违反惩罚 ⭐核心机制

```python
constraint_penalty = 0.0

# 如果动作会导致约束违反（被强制修改）
if psh_info.get('would_violate', False):
    constraint_penalty = -10.0  # 严重惩罚

# 如果实际发生了约束违反
if psh_info.get('is_constraint_violated', False):
    constraint_penalty = -20.0  # 极严重的惩罚

# 如果动作被修改（从违规转为HOLD）
if psh_info.get('action_modified', False):
    constraint_penalty -= 5.0
```

**物理意义**：
- **零容忍策略**：任何约束违反都给予重罚
- 多层惩罚机制：
  - 试图违规：-10
  - 实际违规：-20
  - 动作被修改：-5

**权重**：`w_constraint = -10.0`

---

#### 6. HOLD动作奖励

```python
hold_reward = 0.0
if psh_info.get('action') == 0:  # HOLD动作
    hold_reward = self.w_hold  # w_hold = 0.1
```

**物理意义**：
- 轻微鼓励保持当前状态
- 避免频繁切换动作
- 减少机械磨损

---

#### 7. 不收敛惩罚

```python
convergence_penalty = 0.0 if converged else -1.0
```

**物理意义**：
- 惩罚潮流计算不收敛的情况
- 保证电力系统可求解性

---

### 三、总奖励函数表达式

```python
reward = (
    w_revenue * revenue_norm +           # 收益项
    w_voltage * voltage_penalty +        # 电压惩罚
    w_soc * soc_penalty +                # SOC惩罚
    balance_reward +                     # 平衡奖励
    w_constraint * constraint_penalty +  # 约束惩罚
    hold_reward +                        # HOLD奖励
    convergence_penalty                  # 收敛惩罚
)

# 安全处理
reward = np.clip(reward, -10, 10)
```

**完整公式**：

$$
R = w_r \cdot \frac{Revenue}{5.0} + w_v \cdot \left(-\frac{V_{vio}}{0.02}\right)^2 + w_s \cdot P_{soc} + w_b \cdot (1 - |SOC_u - SOC_l|) + w_c \cdot P_{constraint} + w_h \cdot \mathbb{I}_{HOLD} + P_{conv}
$$

其中：
- $w_r = 1.0$, $w_v = -5.0$, $w_s = -3.0$, $w_b = 0.3$, $w_c = -10.0$, $w_h = 0.1$
- $P_{constraint} \in \{-20, -10, -5, 0\}$ 根据违规程度

---

## 第三部分：PSH约束违反次数降为0的实现机制

### 一、零容忍策略架构

#### 1. 动作安全检查机制

```python
def get_valid_actions(self) -> List[int]:
    """获取当前状态下所有有效的动作"""
    valid_actions = [PSHAction.HOLD]  # HOLD总是有效
    
    # 检查START_GENERATE是否有效
    if self.current_mode == PSHMode.IDLE:
        if (self.upper_soc > self.upper_soc_min + 0.05 and 
            self.lower_soc < self.lower_soc_max - 0.05 and
            self.daily_cycle_count < self.max_daily_cycles):
            valid_actions.append(PSHAction.START_GENERATE)
    
    # 检查START_PUMP是否有效
    if self.current_mode == PSHMode.IDLE:
        if (self.upper_soc < self.upper_soc_max - 0.05 and 
            self.lower_soc > self.lower_soc_min + 0.05 and
            self.daily_cycle_count < self.max_daily_cycles):
            valid_actions.append(PSHAction.START_PUMP)
    
    # 检查STOP是否有效
    if self.current_mode != PSHMode.IDLE:
        if self.operation_duration >= self.min_duration:
            valid_actions.append(PSHAction.STOP)
    
    return valid_actions
```

**关键检查点**：
1. **SOC边界检查**：确保有足够的能量/容量执行动作
2. **运行时长检查**：满足最小运行时长后才能停止
3. **日循环次数检查**：不超过最大日循环次数

---

#### 2. 动作强制转换机制

```python
def step(self, action: int, current_time: int = 0) -> Tuple[float, Dict]:
    action = int(action)
    original_action = action
    
    # 获取当前有效的动作列表
    valid_actions = self.get_valid_actions()
    
    # 检查动作是否有效
    is_valid = action in valid_actions
    
    if not is_valid:
        # 动作无效，强制转为HOLD
        action = PSHAction.HOLD
        action_modified = True
    else:
        action_modified = False
```

**核心逻辑**：
- 任何无效动作都会被强制转为HOLD
- 记录动作是否被修改
- 向智能体反馈动作修改信息

---

#### 3. 功率执行安全检查

```python
# 发电模式安全检查
if self.current_mode == PSHMode.GENERATING:
    target_power = self.rated_gen_power
    energy_required = target_power * self.time_step / self.gen_efficiency
    
    if self.upper_energy - energy_required < self.upper_min:
        # 能量不足，降低功率或停止
        available_energy = self.upper_energy - self.upper_min
        if available_energy > 0:
            target_power = available_energy * self.gen_efficiency / self.time_step
        else:
            target_power = 0.0
            self.current_mode = PSHMode.IDLE
            self.constraint_violations += 1

# 抽水模式安全检查
if self.current_mode == PSHMode.PUMPING:
    target_power = -self.rated_pump_power
    energy_required = abs(target_power) * self.time_step
    
    if self.lower_energy - energy_required < self.lower_min:
        # 容量不足，降低功率或停止
        available_energy = self.lower_energy - self.lower_min
        if available_energy > 0:
            target_power = -available_energy / self.time_step
        else:
            target_power = 0.0
            self.current_mode = PSHMode.IDLE
            self.constraint_violations += 1
```

**双重保护**：
1. **动作级检查**：在执行前检查动作是否可行
2. **功率级检查**：在执行时根据实际能量调整功率

---

### 二、奖励塑形机制

#### 1. 多层惩罚体系

| 违规类型 | 惩罚值 | 说明 |
|---------|--------|------|
| 试图违规（would_violate） | -10 | 智能体选择了会导致违规的动作 |
| 实际违规（is_violated） | -20 | 实际发生了约束违反 |
| 动作被修改 | -5 | 违规动作被强制转为HOLD |

#### 2. 惩罚传递机制

```python
info = {
    'is_constraint_violated': False,  # 实际是否违规
    'action_modified': action_modified,  # 动作是否被修改
    'valid_actions': valid_actions,  # 当前有效动作列表
}
```

智能体通过info获取约束信息，学习避免违规动作。

---

### 三、状态空间设计辅助

```python
def get_state(self) -> np.ndarray:
    # ...其他状态...
    
    # 添加有效动作信息到状态
    valid_actions = self.get_valid_actions()
    can_gen = 1.0 if PSHAction.START_GENERATE in valid_actions else 0.0
    can_pump = 1.0 if PSHAction.START_PUMP in valid_actions else 0.0
    can_stop = 1.0 if PSHAction.STOP in valid_actions else 0.0
    
    return np.array([
        # ...其他状态...
        can_gen,
        can_pump,
        can_stop
    ], dtype=np.float32)
```

**作用**：
- 智能体知道当前哪些动作是有效的
- 帮助智能体学习约束边界
- 加速收敛到安全策略

---

### 四、PPO算法与约束处理的协同

#### 1. 策略学习机制

```python
# PPO策略更新
ratio = torch.exp(log_probs - batch_old_log_probs)
surr1 = ratio * batch_advantages
surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
actor_loss = -torch.min(surr1, surr2).mean()
```

**协同效果**：
- 当智能体选择违规动作时，获得大负奖励
- 优势函数A_t为负，策略更新降低该动作概率
- 裁剪机制保证策略不会突变，稳定学习

#### 2. 熵正则化辅助探索

```python
entropy_loss = -entropy.mean()
loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
```

**作用**：
- 早期高熵鼓励探索，尝试不同动作
- 后期熵减小，策略趋于确定，选择安全动作
- 避免过早收敛到局部最优

---

### 五、实现效果验证

#### 训练结果

| 指标 | 数值 | 说明 |
|------|------|------|
| PSH约束达标率 | **100%** | 200轮全部达标 |
| 平均PSH约束违反 | **0.0次/轮** | 零违规 |
| 奖励改进 | **14.0%** | 前20轮8.12 → 后20轮9.26 |

#### 关键成功因素

1. **零容忍策略**：任何违规动作都被强制转为HOLD
2. **多层惩罚**：试图违规、实际违规、动作修改都有惩罚
3. **状态反馈**：有效动作信息帮助智能体学习约束
4. **PPO稳定性**：裁剪机制保证策略稳定收敛

---

## 总结

### 算法选择结论

**PPO相比DDPG在本项目的优势**：
1. 天然支持离散动作，无需离散化映射
2. 裁剪机制保证训练稳定性
3. 样本复用提高学习效率
4. 内在探索机制更优雅
5. 架构简洁，参数更少

### 奖励函数设计要点

1. **多目标平衡**：收益、电压、SOC、约束多目标加权
2. **二次惩罚**：越限越多惩罚越重
3. **零容忍约束**：违规给予极重惩罚
4. **安全裕度**：SOC边界预留20%缓冲

### PSH零违规实现要点

1. **动作级检查**：执行前验证动作有效性
2. **功率级检查**：执行时根据能量调整功率
3. **强制HOLD**：无效动作强制转为保持
4. **多层惩罚**：不同违规程度不同惩罚
5. **状态反馈**：有效动作信息辅助学习

---

**文档版本**：4.9.2  
**生成时间**：2026-04-09  
**算法**：Proximal Policy Optimization (PPO)
