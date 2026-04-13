"""
IEEE 34节点配电网环境 - 版本4.8.6

优化说明:
1. 超严格的PSH约束控制，目标<7次/轮违反
2. 强化约束违反惩罚机制
3. 改进奖励函数，增加约束违反惩罚权重
4. 添加训练监控指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append('/mnt/okcomputer/output/4.9.1')

from models.pumped_storage import PumpedStorageUnit, BatteryEnergyStorageSystem, PSHMode, PSHAction


class PowerFlowCalculator:
    """
    潮流计算 - 基于功率注入的电压估计
    
    使用改进的线性模型计算电压变化:
    - 基于功率注入和线路阻抗计算电压降落
    - 使用电压灵敏度矩阵
    """

    def __init__(self, node_data: pd.DataFrame, line_data: pd.DataFrame):
        """
        初始化潮流计算器
        
        Args:
            node_data: 节点数据，包含PD, QD, 电压限制等
            line_data: 线路数据，包含R, X等
        """
        self.node_data = node_data
        self.line_data = line_data
        self.n_nodes = len(node_data)
        self.n_lines = len(line_data)
        
        # 提取电压限制
        self.v_max = np.ones(self.n_nodes) * 1.05  # 默认最大电压
        self.v_min = np.ones(self.n_nodes) * 0.95  # 默认最小电压
        
        # 构建节点连接关系
        self._build_node_connections()
        
        # 计算电压灵敏度矩阵
        self._compute_voltage_sensitivity()
        
    def _build_node_connections(self):
        """构建节点连接关系"""
        self.connections = {i: [] for i in range(self.n_nodes)}
        
        for _, line in self.line_data.iterrows():
            from_bus = int(line['FROM']) - 1
            to_bus = int(line['TO']) - 1
            
            r = line['R']
            x = line['X']
            
            self.connections[from_bus].append((to_bus, r, x))
            self.connections[to_bus].append((from_bus, r, x))
    
    def _compute_voltage_sensitivity(self):
        """计算电压对有功和无功功率的灵敏度"""
        # 使用BFS计算每个节点到平衡节点的距离
        visited = [False] * self.n_nodes
        visited[0] = True
        queue = [(0, 0)]  # (node, distance)
        distances = {0: 0}
        
        while queue:
            current, dist = queue.pop(0)
            
            for neighbor, r, x in self.connections[current]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        # 计算灵敏度 - 基于电气距离
        self.voltage_sensitivity_p = np.zeros(self.n_nodes)
        self.voltage_sensitivity_q = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            if i in distances:
                dist = distances[i]
                # 距离越远，灵敏度越高
                # 基础灵敏度 + 距离衰减
                base_sensitivity = 0.002
                distance_factor = 0.001 * dist
                self.voltage_sensitivity_p[i] = base_sensitivity + distance_factor
                self.voltage_sensitivity_q[i] = 2 * (base_sensitivity + distance_factor)
            else:
                # 孤立节点
                self.voltage_sensitivity_p[i] = 0.001
                self.voltage_sensitivity_q[i] = 0.002
    
    def solve(self, P: np.ndarray, Q: np.ndarray, V0: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, np.ndarray, bool]:
        """
        简化交流潮流计算
        
        Args:
            P: 节点有功注入 (MW)
            Q: 节点无功注入 (Mvar)
            V0: 初始电压幅值
            
        Returns:
            V: 电压幅值 (p.u.)
            theta: 电压相角 (rad)
            converged: 是否收敛
        """
        n = self.n_nodes
        
        # 初始化电压
        V = np.ones(n)
        theta = np.zeros(n)
        
        # 平衡节点电压固定
        V[0] = 1.0
        theta[0] = 0.0
        
        # 基于功率注入计算电压变化
        for i in range(1, n):
            # 电压变化 = -灵敏度 * 功率注入
            # 注入功率为正(负荷)，电压下降
            # 注入功率为负(发电)，电压上升
            dV_p = -self.voltage_sensitivity_p[i] * P[i]
            dV_q = -self.voltage_sensitivity_q[i] * Q[i]
            
            V[i] = 1.0 + dV_p + dV_q
        
        # 限制电压范围
        V = np.clip(V, 0.92, 1.08)
        
        converged = True
        
        return V, theta, converged
    
    def check_voltage_violations(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检查电压越限"""
        V = np.clip(V, 0.92, 1.08)
        
        v_upper_violation = V > self.v_max
        v_lower_violation = V < self.v_min
        
        v_violations = np.where(v_upper_violation | v_lower_violation)[0]
        
        v_violation_magnitude = np.zeros(self.n_nodes)
        v_violation_magnitude[v_upper_violation] = V[v_upper_violation] - self.v_max[v_upper_violation]
        v_violation_magnitude[v_lower_violation] = self.v_min[v_lower_violation] - V[v_lower_violation]
        
        return v_violations, v_violation_magnitude


class DistributionNetworkEnv:
    """
    配电网环境 - 版本4.8.6
    
    优化:
    - 超严格的PSH约束控制
    - 强化约束违反惩罚
    - 改进奖励函数
    """
    
    def __init__(
        self,
        node_file: str,
        line_file: str,
        time_series_file: str,
        psh_config: Optional[Dict] = None,
        bess_configs: Optional[List[Dict]] = None,
        time_step: float = 0.25,
        episode_length: int = 672  # 7天 = 672步
    ):
        """
        初始化环境
        
        Args:
            node_file: 节点数据文件路径
            line_file: 线路数据文件路径
            time_series_file: 时间序列数据文件路径
            psh_config: PSH配置
            bess_configs: BESS配置列表
            time_step: 时间步长(小时)
            episode_length: 每轮步数
        """
        # 读取节点和线路数据
        self.node_data = pd.read_csv(node_file)
        self.line_data = pd.read_csv(line_file)
        
        self.n_nodes = len(self.node_data)
        self.time_step = time_step
        self.episode_length = episode_length
        
        # 创建潮流计算器
        self.power_flow = PowerFlowCalculator(self.node_data, self.line_data)
        
        # 读取时间序列数据
        self.time_series_data = pd.read_csv(time_series_file, parse_dates=['date_time'])
        self.time_series_data.set_index('date_time', inplace=True)
        
        # 提取负荷数据 (active_power_node_X)
        self.load_data = self.time_series_data[[col for col in self.time_series_data.columns 
                                                 if col.startswith('active_power_node_')]]
        # 转换为MW (数据是kW)
        self.load_data = self.load_data / 1000.0
        
        # 提取可再生能源数据 (renewable_active_power_node_X)
        self.renewable_data = self.time_series_data[[col for col in self.time_series_data.columns 
                                                      if col.startswith('renewable_active_power_node_')]]
        # 转换为MW (数据是kW)
        self.renewable_data = self.renewable_data / 1000.0
        
        # 处理NaN值
        self.renewable_data = self.renewable_data.fillna(0)
        
        # 提取电价数据
        self.price_data = self.time_series_data['price']
        
        # 保存配置
        self.psh_config_dict = psh_config
        self.bess_configs_list = bess_configs
        
        # 初始化储能系统
        self._init_energy_storages(psh_config, bess_configs)
        
        # 设置状态空间和动作空间
        self._setup_spaces()
        
        # 当前状态
        self.current_step = 0
        self.current_time = 0
        self.episode_start_idx = 0
        
        # 历史记录
        self.voltage_history = []
        self.reward_history = []
        self.action_history = []
        
        # 奖励归一化参数
        self._init_reward_normalization()
        
    def _init_reward_normalization(self):
        """初始化奖励归一化参数 - 版本4.8.6"""
        self.ref_revenue = 5.0  # 参考收益
        self.ref_voltage_violation = 0.02  # 参考电压越限量
        self.ref_soc_violation = 0.05  # 参考SOC越限量
        
        # === 版本4.9.1: 简化奖励权重 ===
        self.w_revenue = 1.0
        self.w_voltage = -5.0
        self.w_soc = -3.0
        self.w_constraint = -10.0
        self.w_balance = 0.3
        self.w_hold = 0.1
        
    def _init_energy_storages(self, psh_config: Optional[Dict], bess_configs: Optional[List[Dict]]):
        """初始化储能系统"""
        # PSH配置 - 版本4.9.1: 根据配电网参数重新设计
        # 净负荷波动约5.5MW，PSH功率设为3.3MW，容量20MWh
        if psh_config is None:
            psh_config = {
                'unit_id': 1,
                'node_id': 34,
                'rated_generation_power': 3.3,  # 3.3MW发电功率
                'rated_pumping_power': 3.3,     # 3.3MW抽水功率
                'upper_reservoir_capacity': 20.0,  # 20MWh容量
                'lower_reservoir_capacity': 20.0,
                'upper_reservoir_min': 2.0,
                'lower_reservoir_min': 2.0,
                'generation_efficiency': 0.88,
                'pumping_efficiency': 0.88,
                'initial_upper_soc': 0.5,
                'initial_lower_soc': 0.5,
                'max_daily_cycles': 6,
                'min_operation_duration': 4,
                'max_operation_duration': 48,
            }
        
        # BESS配置 - 版本4.9.1: 快速调节，功率1.3MW，容量2.6MWh
        if bess_configs is None:
            bess_configs = [
                {
                    'unit_id': 2,
                    'node_id': 16,
                    'max_power': 1.3,  # 1.3MW
                    'capacity': 2.6,   # 2.6MWh
                    'min_soc': 0.1,
                    'max_soc': 0.9,
                    'charge_efficiency': 0.95,
                    'discharge_efficiency': 0.95,
                    'initial_soc': 0.5,
                    'ramp_rate_limit': 0.5
                },
                {
                    'unit_id': 3,
                    'node_id': 27,
                    'max_power': 1.3,  # 1.3MW
                    'capacity': 2.6,   # 2.6MWh
                    'min_soc': 0.1,
                    'max_soc': 0.9,
                    'charge_efficiency': 0.95,
                    'discharge_efficiency': 0.95,
                    'initial_soc': 0.5,
                    'ramp_rate_limit': 0.5
                }
            ]
        
        self.psh = PumpedStorageUnit(**psh_config, time_step=self.time_step)
        self.bess_units = []
        for config in bess_configs:
            self.bess_units.append(BatteryEnergyStorageSystem(**config, time_step=self.time_step))
        
        self.n_bess = len(self.bess_units)
        
    def _setup_spaces(self):
        """设置状态空间和动作空间"""
        # 状态维度: 34节点净负荷 + 电价 + PSH状态(9维) + 2*BESS状态(各2维) + 时间
        self.state_dim = 34 + 1 + 9 + 2 * 2 + 1
        
        # 动作维度: 3 (PSH + BESS1 + BESS2)
        self.action_dim = 3
        
    def _discretize_psh_action(self, continuous_action: float) -> int:
        """
        将PSH连续动作离散化 - 改进映射
        
        映射规则:
        - [-1.0, -0.6) -> 3 (STOP)
        - [-0.6, -0.2) -> 0 (HOLD)
        - [-0.2, 0.2)  -> 1 (START_GENERATE)
        - [0.2, 0.6)   -> 2 (START_PUMP)
        - [0.6, 1.0]   -> 0 (HOLD)
        """
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
    
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """
        重置环境 - 每轮结束后重置PSH和EBSS的SOC为0.5
        """
        if start_idx is None:
            # 随机抽取连续7天的起始索引
            max_start = len(self.load_data) - self.episode_length - 1
            if max_start > 0:
                self.episode_start_idx = np.random.randint(0, max_start)
            else:
                self.episode_start_idx = 0
        else:
            self.episode_start_idx = start_idx
        
        self.current_step = 0
        self.current_time = self.episode_start_idx
        
        # 重置PSH状态 - SOC重置为0.5
        self.psh.reset()
        
        # 重置BESS状态 - SOC重置为0.5
        for bess in self.bess_units:
            bess.reset()
        
        # 清空历史记录
        self.voltage_history = []
        self.reward_history = []
        self.action_history = []
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: [psh_action, bess1_action, bess2_action]
            
        Returns:
            next_state: 下一状态
            reward: 奖励 (平均奖励)
            done: 是否结束
            info: 额外信息
        """
        # 处理PPO的离散动作输出
        if np.isscalar(action):
            # PPO输出的是单个整数动作索引
            psh_discrete = int(action) % 4  # 映射到0-3
            # BESS动作为0（保持）
            bess1_action = 0.0
            bess2_action = 0.0
        else:
            # 连续动作（兼容旧版本）
            action = np.clip(action, -1.0, 1.0)
            psh_continuous = action[0]
            psh_discrete = self._discretize_psh_action(psh_continuous)
            bess1_action = action[1] if len(action) > 1 else 0.0
            bess2_action = action[2] if len(action) > 2 else 0.0
        
        self.action_history.append(np.array([psh_discrete, bess1_action, bess2_action]))
        
        # 执行PSH动作 - 传递当前时间步
        psh_power, psh_info = self.psh.step(psh_discrete, self.current_step)
        psh_info['action'] = psh_discrete
        
        # 执行BESS动作 (连续)
        bess_powers = []
        bess_socs = []
        bess_actions = [bess1_action, bess2_action]
        for i, bess in enumerate(self.bess_units):
            power, soc, info = bess.step(bess_actions[i])
            bess_powers.append(power)
            bess_socs.append(soc)
        
        # 获取当前时刻的负荷和可再生能源数据
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes]
        renewable_p = self.renewable_data.iloc[self.current_time].values[:self.n_nodes]
        price = self.price_data.iloc[self.current_time]
        
        # 计算净负荷 (负荷 - 可再生能源)
        net_load = load_p - renewable_p
        
        # 更新节点注入功率
        node_p = net_load.copy()
        node_q = self.node_data['QD'].values[:self.n_nodes] / 1000.0  # kW -> MW
        
        # PSH功率注入 (节点34)
        psh_node_idx = self.psh.node_id - 1
        if psh_node_idx < self.n_nodes:
            node_p[psh_node_idx] -= psh_power
        
        # BESS功率注入 (节点16和27)
        for bess, power in zip(self.bess_units, bess_powers):
            bess_node_idx = bess.node_id - 1
            if bess_node_idx < self.n_nodes:
                node_p[bess_node_idx] -= power
        
        # 运行潮流计算
        V, theta, converged = self.power_flow.solve(node_p, node_q)
        
        # 检查电压越限
        v_violations, v_violation_mag = self.power_flow.check_voltage_violations(V)
        
        # 计算奖励 (平均奖励)
        reward = self._calculate_reward(
            price, psh_power, bess_powers,
            v_violation_mag, converged,
            psh_info, bess_socs
        )
        
        # 记录历史
        self.voltage_history.append(V.copy())
        self.reward_history.append(reward)
        
        # 更新步数
        self.current_step += 1
        self.current_time += 1
        done = self.current_step >= self.episode_length
        
        next_state = self._get_state()
        
        info = {
            'voltage': V,
            'v_violations': v_violations,
            'v_violation_magnitude': np.sum(v_violation_mag),
            'v_violation_count': len(v_violations),
            'psh_power': psh_power,
            'psh_action': psh_discrete,
            'psh_upper_soc': psh_info['upper_soc'],
            'psh_lower_soc': psh_info['lower_soc'],
            'psh_mode': psh_info['mode'],
            'psh_constraint_violated': psh_info.get('is_constraint_violated', False),
            'psh_action_modified': psh_info.get('action_modified', False),
            'bess_powers': bess_powers,
            'bess_socs': bess_socs,
            'converged': converged,
            'price': price,
            'voltage_mean': np.mean(V),
            'voltage_std': np.std(V),
            'voltage_min': np.min(V),
            'voltage_max': np.max(V),
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 获取当前时刻的负荷和可再生能源数据
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes]
        renewable_p = self.renewable_data.iloc[self.current_time].values[:self.n_nodes]
        
        # 净负荷
        net_load = load_p - renewable_p
        
        # 电价
        price = self.price_data.iloc[self.current_time]
        
        # PSH状态 (6维)
        psh_state = self.psh.get_state()
        
        # BESS状态 (各2维)
        bess_states = []
        for bess in self.bess_units:
            bess_states.extend(bess.get_state())
        
        # 时间特征 (0-1)
        hour = (self.current_time % 96) / 96.0
        
        # 归一化
        net_load_norm = net_load / 10.0  # 假设最大负荷10MW
        net_load_norm = np.clip(net_load_norm, -5, 5)
        price_norm = price / 100.0  # 假设最大电价100
        
        state = np.concatenate([
            net_load_norm.astype(np.float32),
            [price_norm],
            psh_state,
            bess_states,
            [hour]
        ])
        
        return state
    
    def _calculate_reward(
        self,
        price: float,
        psh_power: float,
        bess_powers: List[float],
        v_violation_mag: np.ndarray,
        converged: bool,
        psh_info: Dict,
        bess_socs: List[float]
    ) -> float:
        """
        计算奖励 (平均奖励) - 版本4.8.6，强化约束惩罚
        """
        # 1. 能量套利收益
        psh_revenue = price * psh_power * self.time_step
        bess_revenue = sum(price * power * self.time_step for power in bess_powers)
        total_revenue = psh_revenue + bess_revenue
        
        # 避免除以0
        revenue_norm = total_revenue / self.ref_revenue if self.ref_revenue > 0 else 0.0
        
        # 2. 电压越限惩罚
        voltage_violation_sum = np.sum(v_violation_mag)
        voltage_penalty = -(voltage_violation_sum / self.ref_voltage_violation) ** 2 if self.ref_voltage_violation > 0 else 0.0
        
        # 3. SOC约束惩罚
        soc_penalty = 0.0
        
        # PSH SOC约束
        psh_upper_soc = psh_info['upper_soc']
        psh_lower_soc = psh_info['lower_soc']
        
        # 上水库SOC边界惩罚 - 软边界
        if psh_upper_soc > 0.80:
            soc_penalty -= ((psh_upper_soc - 0.80) / self.ref_soc_violation) ** 2
        elif psh_upper_soc < 0.20:
            soc_penalty -= ((0.20 - psh_upper_soc) / self.ref_soc_violation) ** 2
        
        # 下水库SOC边界惩罚
        if psh_lower_soc > 0.80:
            soc_penalty -= ((psh_lower_soc - 0.80) / self.ref_soc_violation) ** 2
        elif psh_lower_soc < 0.20:
            soc_penalty -= ((0.20 - psh_lower_soc) / self.ref_soc_violation) ** 2
        
        # 4. SOC平衡奖励 - 鼓励上下水库SOC保持平衡
        soc_balance = 1.0 - abs(psh_upper_soc - psh_lower_soc)
        balance_reward = self.w_balance * soc_balance
        
        # === 版本4.8.7: 强化PSH约束违反惩罚 ===
        constraint_penalty = 0.0
        
        # 如果动作会导致约束违反（被强制修改）
        if psh_info.get('would_violate', False):
            constraint_penalty = -10.0  # 严重惩罚试图违反约束的动作
        
        # 如果实际发生了约束违反
        if psh_info.get('is_constraint_violated', False):
            constraint_penalty = -20.0  # 极严重的惩罚
        
        # 如果动作被修改（从违规转为HOLD）
        if psh_info.get('action_modified', False):
            constraint_penalty -= 5.0
        
        # 6. HOLD动作奖励 - 鼓励保守策略
        hold_reward = 0.0
        if psh_info.get('action') == 0:  # HOLD动作
            hold_reward = self.w_hold
        
        # 7. 不收敛惩罚
        convergence_penalty = 0.0 if converged else -1.0
        
        # 总奖励 (加权组合)
        reward = (
            self.w_revenue * revenue_norm +
            self.w_voltage * voltage_penalty +
            self.w_soc * soc_penalty +
            balance_reward +
            self.w_constraint * constraint_penalty +
            hold_reward +
            convergence_penalty
        )
        
        # 检查nan
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        # 裁剪 - 限制奖励范围
        reward = np.clip(reward, -10, 10)
        
        return float(reward)
    
    def get_storage_states(self) -> Dict:
        """获取所有储能系统的状态"""
        states = {
            'psh': {
                'upper_soc': self.psh.upper_soc,
                'lower_soc': self.psh.lower_soc,
                'power': self.psh.current_power,
                'mode': self.psh.current_mode,
                'daily_cycles': self.psh.daily_cycle_count
            },
            'bess': []
        }
        for bess in self.bess_units:
            states['bess'].append({
                'soc': bess.current_soc,
                'power': bess.current_power
            })
        return states
