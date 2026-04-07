"""
IEEE 34节点配电网环境 - 方案2: 统一DRL框架，混合动作空间

关键修改:
1. 混合动作空间: PSH离散 + BESS连续
2. PSH节点改为34，BESS节点改为16和27
3. PSH定速运行，上下水库分别建模
4. 单一DRL智能体统一决策

修复点:
1. 上下水库关联逻辑 - 发电时上库水减少，下库增加；抽水时相反
2. 潮流计算 - 使用pandapower确保准确性和收敛性
3. 奖励函数 - 各分量归一化，避免数量级差异
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces
import sys
import warnings

try:
    import pandapower as pp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    print("警告: pandapower未安装，将使用简化潮流计算")

sys.path.append('/mnt/okcomputer/output/psh_scheme2')

from models.pumped_storage import PumpedStorageUnit, BatteryEnergyStorageSystem, PSHMode, PSHAction


class PowerFlowCalculator:
    """
    潮流计算 - 使用pandapower确保准确性
    
    修复点:
    - 使用pandapower库进行准确的潮流计算
    - 正确的线路参数配置
    """

    def __init__(self, node_data: pd.DataFrame, line_data: pd.DataFrame):
        self.node_data = node_data
        self.line_data = line_data
        self.n_nodes = len(node_data)
        self.n_lines = len(line_data)

        self.v_max = node_data['Vmax'].values
        self.v_min = node_data['Vmin'].values
        
        self.base_mva = 1.0
        
        # 创建pandapower网络
        if PANDAPOWER_AVAILABLE:
            self.net = self._create_pandapower_network()
        else:
            self.net = None

    def _create_pandapower_network(self):
        """创建pandapower网络"""
        net = pp.create_empty_network()
        
        # 创建总线
        for _, node in self.node_data.iterrows():
            bus_type = 'b' if node['type'] == 2 else 'n'
            pp.create_bus(net, 
                         vn_kv=12.66,  # 配电网电压等级
                         name=f"Bus {node['bus_id']}",
                         type=bus_type)
        
        # 创建外部电网（slack bus）
        pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0)
        
        # 创建线路 - 使用正确的参数转换
        # 原始数据是标幺值，需要转换为实际欧姆值
        base_voltage = 12.66  # kV
        base_impedance = base_voltage**2 / self.base_mva  # 欧姆
        
        for _, line in self.line_data.iterrows():
            from_bus = int(line['from_bus']) - 1
            to_bus = int(line['to_bus']) - 1
            
            # 假设线路长度为1km，将标幺值阻抗转换为欧姆/km
            r_ohm_per_km = line['r'] * base_impedance
            x_ohm_per_km = line['x'] * base_impedance
            
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=1.0,
                r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=x_ohm_per_km,
                c_nf_per_km=0.0,
                max_i_ka=10.0
            )
        
        return net

    def solve(self, P: np.ndarray, Q: np.ndarray, V0: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        潮流计算
        
        修复点:
        - 使用pandapower进行准确的潮流计算
        """
        if PANDAPOWER_AVAILABLE and self.net is not None:
            return self._solve_pandapower(P, Q)
        else:
            return self._solve_simplified(P, Q)

    def _solve_pandapower(self, P: np.ndarray, Q: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, bool]:
        """使用pandapower求解"""
        try:
            # 清除旧的负荷和发电
            if len(self.net.load) > 0:
                self.net.load.drop(self.net.load.index, inplace=True)
            if len(self.net.sgen) > 0:
                self.net.sgen.drop(self.net.sgen.index, inplace=True)
            
            # 添加新的负荷和发电
            for i in range(1, self.n_nodes):  # 跳过slack bus
                if P[i] > 0:  # 负荷
                    pp.create_load(self.net, 
                                 bus=i, 
                                 p_mw=P[i], 
                                 q_mvar=Q[i],
                                 name=f"Load {i+1}")
                elif P[i] < 0:  # 发电（负的负荷）
                    pp.create_sgen(self.net,
                                 bus=i,
                                 p_mw=-P[i],
                                 q_mvar=-Q[i],
                                 name=f"Gen {i+1}")
            
            # 运行潮流计算
            pp.runpp(self.net, algorithm='nr', init='flat', max_iteration=100)
            
            # 提取结果
            V = self.net.res_bus.vm_pu.values
            theta = np.radians(self.net.res_bus.va_degree.values)
            
            # 计算支路功率
            S_branch = np.zeros(self.n_lines, dtype=complex)
            for idx in range(min(self.n_lines, len(self.net.res_line))):
                p_from = self.net.res_line.p_from_mw.values[idx]
                q_from = self.net.res_line.q_from_mvar.values[idx]
                S_branch[idx] = complex(p_from, q_from)
            
            converged = True
            
        except Exception as e:
            # 如果pandapower失败，使用简化方法
            V = self._solve_simplified(P, Q)[0]
            theta = np.zeros(self.n_nodes)
            S_branch = np.zeros(self.n_lines, dtype=complex)
            converged = False
        
        return V, theta, S_branch, converged

    def _solve_simplified(self, P: np.ndarray, Q: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, bool]:
        """简化潮流计算 - 基于负荷的电压估计"""
        V = np.ones(self.n_nodes)
        
        # 根据注入功率调整电压
        for i in range(1, self.n_nodes):
            if P[i] > 0:  # 负荷
                # 负荷越大，电压越低
                V[i] = 1.0 - 0.02 * min(P[i] / 2.0, 0.1)
            elif P[i] < 0:  # 发电
                # 发电越大，电压越高
                V[i] = 1.0 + 0.01 * min(-P[i] / 2.0, 0.05)
        
        theta = np.zeros(self.n_nodes)
        S_branch = np.zeros(self.n_lines, dtype=complex)
        
        return V, theta, S_branch, True

    def check_voltage_violations(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检查电压越限"""
        V = np.clip(V, 0.5, 1.5)

        v_upper_violation = V > self.v_max
        v_lower_violation = V < self.v_min

        v_violations = np.where(v_upper_violation | v_lower_violation)[0]

        v_violation_magnitude = np.zeros(self.n_nodes)
        v_violation_magnitude[v_upper_violation] = V[v_upper_violation] - self.v_max[v_upper_violation]
        v_violation_magnitude[v_lower_violation] = self.v_min[v_lower_violation] - V[v_lower_violation]

        return v_violations, v_violation_magnitude


class DistributionNetworkEnv(gym.Env):
    """
    配电网环境 - 方案2: 统一DRL框架，混合动作空间
    
    动作空间:
    - action[0]: PSH离散动作 (连续值[-1,1]映射到 {0:保持, 1:启动发电, 2:启动抽水, 3:停止})
    - action[1]: BESS1连续动作 [-1, 1]
    - action[2]: BESS2连续动作 [-1, 1]
    """

    def __init__(
            self,
            node_file: str,
            line_file: str,
            load_data: pd.DataFrame,
            price_data: pd.DataFrame,
            pv_data: Optional[pd.DataFrame] = None,
            psh_config: Optional[Dict] = None,
            bess_configs: Optional[List[Dict]] = None,
            time_step: float = 0.25,
            episode_length: int = 96
    ):
        super().__init__()

        self.node_data = pd.read_csv(node_file)
        self.line_data = pd.read_csv(line_file)

        self.n_nodes = len(self.node_data)
        self.time_step = time_step
        self.episode_length = episode_length

        self._validate_network_data()
        self.power_flow = PowerFlowCalculator(self.node_data, self.line_data)

        self.load_data = load_data
        self.price_data = price_data
        self.pv_data = pv_data if pv_data is not None else pd.DataFrame()

        self._check_data_scale()
        self._init_energy_storages(psh_config, bess_configs)
        self._setup_spaces()

        self.current_step = 0
        self.current_time = 0
        self.episode_start_idx = 0

        self.voltage_history = []
        self.reward_history = []
        self.action_history = []

        self.max_load = 500.0
        self.max_price = 100.0

        self.last_action = np.zeros(self.action_dim)
        self.emergency_count = 0
        
        # 奖励归一化参数
        self._init_reward_normalization()

    def _init_reward_normalization(self):
        """
        初始化奖励归一化参数
        
        修复点:
        - 各奖励分量归一化到相近的数量级
        - 使用参考值进行归一化
        """
        # 参考值用于归一化
        self.ref_revenue = 50.0  # 参考收益 (¥/step)
        self.ref_voltage_violation = 0.05  # 参考电压越限量 (p.u.)
        self.ref_soc_violation = 0.1  # 参考SOC越限量
        self.ref_action_change = 0.5  # 参考动作变化量
        
        # 各分量权重（归一化后）
        self.w_revenue = 1.0
        self.w_voltage = -1.0
        self.w_convergence = -0.5
        self.w_soc = -0.3
        self.w_smooth = -0.1

    def _validate_network_data(self):
        """验证网络数据合理性"""
        for _, line in self.line_data.iterrows():
            r, x = line['r'], line['x']
            if r < 0 or x < 0:
                warnings.warn(f"线路 {line['from_bus']}-{line['to_bus']} 阻抗为负！")
            if r > 10 or x > 10:
                warnings.warn(f"线路 {line['from_bus']}-{line['to_bus']} 阻抗过大")

    def _check_data_scale(self):
        """自动检测并转换数据单位"""
        load_max = self.load_data.values.max()
        if load_max > 1000:
            print(f"检测到负荷数据单位为kW，自动转换为MW")
            self.load_data = self.load_data / 1000.0

        if not self.pv_data.empty:
            pv_max = self.pv_data.values.max()
            if pv_max > 1000:
                print(f"检测到光伏数据单位为kW，自动转换为MW")
                self.pv_data = self.pv_data / 1000.0

    def _init_energy_storages(self, psh_config: Optional[Dict], bess_configs: Optional[List[Dict]]):
        """初始化储能系统"""
        # PSH配置 - 节点34，定速运行，上下水库
        if psh_config is None:
            psh_config = {
                'unit_id': 1,
                'node_id': 34,
                'rated_generation_power': 5.0,
                'rated_pumping_power': 5.0,
                'upper_reservoir_capacity': 30.0,
                'lower_reservoir_capacity': 30.0,
                'upper_reservoir_min': 3.0,
                'lower_reservoir_min': 3.0,
                'generation_efficiency': 0.85,
                'pumping_efficiency': 0.85,
                'initial_upper_soc': 0.5,
                'initial_lower_soc': 0.5,
                'max_daily_cycles': 4,
                'min_operation_duration': 8,
                'max_operation_duration': 24,
            }

        # BESS配置 - 节点16和27
        if bess_configs is None:
            bess_configs = [
                {
                    'unit_id': 2,
                    'node_id': 16,
                    'max_power': 2.0,
                    'capacity': 4.0,
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                    'charge_efficiency': 0.95,
                    'discharge_efficiency': 0.95,
                    'initial_soc': 0.5,
                    'ramp_rate_limit': 1.0
                },
                {
                    'unit_id': 3,
                    'node_id': 27,
                    'max_power': 2.0,
                    'capacity': 4.0,
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                    'charge_efficiency': 0.95,
                    'discharge_efficiency': 0.95,
                    'initial_soc': 0.5,
                    'ramp_rate_limit': 1.0
                }
            ]

        self.psh = PumpedStorageUnit(**psh_config, time_step=self.time_step)
        self.bess_units = []
        for config in bess_configs:
            self.bess_units.append(BatteryEnergyStorageSystem(**config, time_step=self.time_step))

        self.n_bess = len(self.bess_units)

    def _setup_spaces(self):
        """设置状态空间和动作空间"""
        # 状态维度: 34节点负荷 + 电价 + PSH状态(6维) + 2*BESS状态(各2维) + 时间
        self.state_dim = 46
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # 动作维度: 3 (PSH + BESS1 + BESS2)
        self.action_dim = 3
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

    def _discretize_psh_action(self, continuous_action: float) -> int:
        """
        将PSH连续动作离散化
        
        映射规则:
        - [-1.0, -0.5) -> 3 (STOP)
        - [-0.5, 0.0)  -> 0 (HOLD)
        - [0.0, 0.5)   -> 1 (START_GENERATE)
        - [0.5, 1.0]   -> 2 (START_PUMP)
        """
        if continuous_action < -0.5:
            return PSHAction.STOP
        elif continuous_action < 0.0:
            return PSHAction.HOLD
        elif continuous_action < 0.5:
            return PSHAction.START_GENERATE
        else:
            return PSHAction.START_PUMP

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """重置环境"""
        if start_idx is None:
            max_start = len(self.load_data) - self.episode_length - 1
            self.episode_start_idx = np.random.randint(0, max(max_start, 1))
        else:
            self.episode_start_idx = start_idx

        self.current_step = 0
        self.current_time = self.episode_start_idx

        self.psh.reset()
        for bess in self.bess_units:
            bess.reset()

        self.voltage_history = []
        self.reward_history = []
        self.action_history = []
        self.last_action = np.zeros(self.action_dim)
        self.emergency_count = 0

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作 - 混合动作空间
        
        Args:
            action: [psh_action, bess1_action, bess2_action]
                   psh_action: 连续值[-1,1]，映射到离散动作
                   bess1_action, bess2_action: 连续值[-1,1]
        """
        action = np.clip(action, -1.0, 1.0)

        # 紧急制动
        if self.emergency_count > 0:
            action = action * 0.1
            self.emergency_count -= 1

        self.action_history.append(action.copy())

        # 解析PSH动作 (连续->离散)
        psh_continuous = action[0]
        psh_discrete = self._discretize_psh_action(psh_continuous)
        
        # 执行PSH动作
        psh_power, psh_info = self.psh.step(psh_discrete)
        psh_info['action'] = psh_discrete
        psh_info['continuous_action'] = psh_continuous
        
        # 执行BESS动作 (连续)
        bess_powers = []
        bess_socs = []
        for i, bess in enumerate(self.bess_units):
            power, soc, info = bess.step(action[i + 1])
            bess_powers.append(power)
            bess_socs.append(soc)

        # 获取负荷数据
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes]
        price = self.price_data.iloc[self.current_time]['price']

        # 光伏出力
        if not self.pv_data.empty:
            pv_p = self.pv_data.iloc[self.current_time].values[:self.n_nodes]
            net_load = load_p - pv_p
        else:
            net_load = load_p

        # 更新节点注入功率
        node_p = net_load.copy()

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
        node_q = np.zeros(self.n_nodes)
        V, theta, S_branch, converged = self.power_flow.solve(node_p, node_q)

        # 检查电压合理性
        voltage_reasonable = np.all((V > 0.5) & (V < 1.5))
        if not voltage_reasonable or not converged:
            self.emergency_count = 5
            V = np.clip(V, 0.5, 1.5)

        # 检查电压越限
        v_violations, v_violation_mag = self.power_flow.check_voltage_violations(V)

        # 计算奖励（使用归一化版本）
        reward = self._calculate_reward_normalized(
            price, psh_power, bess_powers,
            v_violation_mag, converged and voltage_reasonable,
            psh_info['upper_soc'], psh_info['lower_soc'], 
            bess_socs, action
        )

        self.voltage_history.append(V.copy())
        self.reward_history.append(reward)
        self.last_action = action.copy()

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
            'psh_continuous_action': psh_continuous,
            'psh_upper_soc': psh_info['upper_soc'],
            'psh_lower_soc': psh_info['lower_soc'],
            'psh_mode': psh_info['mode'],
            'psh_daily_cycles': psh_info['daily_cycles'],
            'bess_powers': bess_powers,
            'bess_socs': bess_socs,
            'converged': converged and voltage_reasonable,
            'price': price,
            'voltage_reasonable': voltage_reasonable,
            'voltage_mean': np.mean(V),
            'voltage_std': np.std(V)
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes]

        if not self.pv_data.empty:
            pv_p = self.pv_data.iloc[self.current_time].values[:self.n_nodes]
            net_load = load_p - pv_p
        else:
            net_load = load_p

        price = self.price_data.iloc[self.current_time]['price']

        # PSH状态 (6维)
        psh_state = self.psh.get_state()
        
        # BESS状态 (各2维)
        bess_states = []
        for bess in self.bess_units:
            bess_states.extend(bess.get_state())

        hour = (self.current_time % 96) / 96.0

        # 归一化
        max_load_mw = self.max_load / 1000.0
        net_load_norm = net_load / max_load_mw
        net_load_norm = np.clip(net_load_norm, -5, 5)
        price_norm = price / self.max_price

        state = np.concatenate([
            net_load_norm.astype(np.float32),
            [price_norm],
            psh_state,
            bess_states,
            [hour]
        ])

        return state

    def _calculate_reward_normalized(
            self,
            price: float,
            psh_power: float,
            bess_powers: List[float],
            v_violation_mag: np.ndarray,
            converged: bool,
            psh_upper_soc: float,
            psh_lower_soc: float,
            bess_socs: List[float],
            action: np.ndarray
    ) -> float:
        """
        计算归一化奖励
        
        修复点:
        - 各分量归一化到相近的数量级
        - 使用参考值进行归一化，避免数量级差异
        """

        # 1. 能量套利收益 (归一化)
        psh_revenue = price * psh_power * self.time_step
        bess_revenue = sum(price * power * self.time_step for power in bess_powers)
        total_revenue = psh_revenue + bess_revenue
        # 归一化到 [-1, 1] 范围
        revenue_norm = np.clip(total_revenue / self.ref_revenue, -5, 5)

        # 2. 电压越限惩罚 (归一化)
        voltage_violation_sum = np.sum(v_violation_mag)
        voltage_penalty = -(voltage_violation_sum / self.ref_voltage_violation) ** 2
        voltage_penalty = np.clip(voltage_penalty, -10, 0)

        # 3. 不收敛惩罚
        convergence_penalty = 0.0 if converged else -1.0

        # 4. SOC软约束 (归一化)
        soc_penalty = 0.0
        
        # PSH上水库
        if psh_upper_soc < 0.2:
            soc_penalty -= ((0.2 - psh_upper_soc) / self.ref_soc_violation) ** 2
        elif psh_upper_soc > 0.9:
            soc_penalty -= ((psh_upper_soc - 0.9) / self.ref_soc_violation) ** 2
            
        # PSH下水库
        if psh_lower_soc < 0.2:
            soc_penalty -= ((0.2 - psh_lower_soc) / self.ref_soc_violation) ** 2
        elif psh_lower_soc > 0.9:
            soc_penalty -= ((psh_lower_soc - 0.9) / self.ref_soc_violation) ** 2

        # BESS SOC
        for soc in bess_socs:
            if soc < 0.25:
                soc_penalty -= ((0.25 - soc) / self.ref_soc_violation) ** 2
            elif soc > 0.85:
                soc_penalty -= ((soc - 0.85) / self.ref_soc_violation) ** 2
        
        soc_penalty = np.clip(soc_penalty, -5, 0)

        # 5. 动作平滑性 (归一化)
        action_change = np.sum((action - self.last_action) ** 2)
        smooth_penalty = -action_change / (self.ref_action_change ** 2)
        smooth_penalty = np.clip(smooth_penalty, -2, 0)

        # 总奖励 (加权组合)
        reward = (
            self.w_revenue * revenue_norm +
            self.w_voltage * voltage_penalty +
            self.w_convergence * convergence_penalty +
            self.w_soc * soc_penalty +
            self.w_smooth * smooth_penalty
        )

        # 最终裁剪
        reward = np.clip(reward, -10, 10)

        return float(reward)

    def render(self, mode='human'):
        """渲染"""
        if len(self.voltage_history) == 0:
            return
        latest_v = self.voltage_history[-1]
        print(f"Step: {self.current_step}, Voltage: [{np.min(latest_v):.4f}, {np.max(latest_v):.4f}], "
              f"PSH Upper SOC: {self.psh.upper_soc:.3f}, Lower SOC: {self.psh.lower_soc:.3f}, "
              f"Reward: {self.reward_history[-1]:.2f}")

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
