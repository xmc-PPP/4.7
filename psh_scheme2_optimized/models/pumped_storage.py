"""
抽水储能机组(Pumped Storage Hydro, PSH)建模 - 方案2: 离散动作控制

关键修改:
1. 定速运行模式 (发电功率固定、抽水功率固定)
2. 增加上下水库分别建模
3. 支持离散动作输入: {0:保持, 1:启动发电, 2:启动抽水, 3:停止}
4. 每日运行次数限制 (0-4次)
5. 每次运行时长限制 (2-6小时)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from enum import IntEnum


class PSHMode(IntEnum):
    """PSH运行模式"""
    IDLE = 0          # 停机
    GENERATING = 1    # 发电模式
    PUMPING = 2       # 抽水模式


class PSHAction(IntEnum):
    """PSH离散动作"""
    HOLD = 0          # 保持当前状态
    START_GENERATE = 1  # 启动发电
    START_PUMP = 2      # 启动抽水
    STOP = 3            # 停止运行


class PumpedStorageUnit:
    """
    定速抽水储能机组模型 - 离散动作控制版本
    
    特征:
    - 定速运行: 发电功率固定、抽水功率固定
    - 上下水库分别建模
    - 离散动作控制: {保持, 启动发电, 启动抽水, 停止}
    - 每日启停次数限制
    - 运行时长跟踪
    """
    
    def __init__(
        self,
        unit_id: int,
        node_id: int,
        rated_generation_power: float,   # 额定发电功率 (MW) - 定速
        rated_pumping_power: float,      # 额定抽水功率 (MW) - 定速
        upper_reservoir_capacity: float, # 上水库容量 (MWh)
        lower_reservoir_capacity: float, # 下水库容量 (MWh)
        upper_reservoir_min: float,      # 上水库最小蓄能 (MWh)
        lower_reservoir_min: float,      # 下水库最小蓄能 (MWh)
        generation_efficiency: float,    # 发电效率 (0-1)
        pumping_efficiency: float,      # 抽水效率 (0-1)
        initial_upper_soc: float,       # 上水库初始SOC (0-1)
        initial_lower_soc: float,       # 下水库初始SOC (0-1)
        max_daily_cycles: int = 4,      # 最大日启停次数
        min_operation_duration: int = 8,   # 最短运行时长 (15分钟步数, 8=2小时)
        max_operation_duration: int = 24,  # 最长运行时长 (15分钟步数, 24=6小时)
        time_step: float = 0.25          # 时间步长 (小时)
    ):
        self.unit_id = unit_id
        self.node_id = node_id
        self.rated_gen_power = rated_generation_power
        self.rated_pump_power = rated_pumping_power
        self.upper_capacity = upper_reservoir_capacity
        self.lower_capacity = lower_reservoir_capacity
        self.upper_min = upper_reservoir_min
        self.lower_min = lower_reservoir_min
        self.gen_efficiency = generation_efficiency
        self.pump_efficiency = pumping_efficiency
        self.max_daily_cycles = max_daily_cycles
        self.min_duration = min_operation_duration
        self.max_duration = max_operation_duration
        self.time_step = time_step
        
        # 当前状态
        self.current_mode = PSHMode.IDLE
        self.operation_start_time = None
        self.operation_duration = 0
        self.daily_cycle_count = 0
        
        # 水库状态
        self.upper_energy = initial_upper_soc * (upper_reservoir_capacity - upper_reservoir_min) + upper_reservoir_min
        self.lower_energy = initial_lower_soc * (lower_reservoir_capacity - lower_reservoir_min) + lower_reservoir_min
        
        # 当前功率输出
        self.current_power = 0.0
        
        # 历史记录
        self.upper_soc_history = [self.upper_soc]
        self.lower_soc_history = [self.lower_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]
        
    @property
    def upper_soc(self) -> float:
        """上水库SOC"""
        return (self.upper_energy - self.upper_min) / (self.upper_capacity - self.upper_min)
    
    @property
    def lower_soc(self) -> float:
        """下水库SOC"""
        return (self.lower_energy - self.lower_min) / (self.lower_capacity - self.lower_min)
    
    def reset(self, initial_upper_soc: Optional[float] = None, initial_lower_soc: Optional[float] = None):
        """重置机组状态"""
        if initial_upper_soc is not None:
            self.upper_energy = initial_upper_soc * (self.upper_capacity - self.upper_min) + self.upper_min
        if initial_lower_soc is not None:
            self.lower_energy = initial_lower_soc * (self.lower_capacity - self.lower_min) + self.lower_min
            
        self.current_mode = PSHMode.IDLE
        self.operation_start_time = None
        self.operation_duration = 0
        self.daily_cycle_count = 0
        self.current_power = 0.0
        
        self.upper_soc_history = [self.upper_soc]
        self.lower_soc_history = [self.lower_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]
        
    def _can_generate(self) -> bool:
        """检查是否可以发电"""
        min_energy_required = self.rated_gen_power * self.time_step / self.gen_efficiency
        return (self.upper_energy - self.upper_min) >= min_energy_required * 2
        
    def _can_pump(self) -> bool:
        """检查是否可以抽水"""
        max_energy_can_store = (self.lower_capacity - self.lower_energy) * self.pump_efficiency
        energy_to_store = self.rated_pump_power * self.time_step * self.pump_efficiency
        return max_energy_can_store >= energy_to_store * 2
        
    def _check_duration_constraint(self) -> bool:
        """检查运行时长约束"""
        if self.current_mode == PSHMode.IDLE:
            return True
        if self.operation_duration < self.min_duration:
            return False
        if self.operation_duration >= self.max_duration:
            return True
        return True
        
    def step(self, action: int) -> Tuple[float, Dict]:
        """
        执行一个时间步的动作 - 离散动作控制
        
        Args:
            action: 离散动作 (0=保持, 1=启动发电, 2=启动抽水, 3=停止)
        
        Returns:
            actual_power: 实际输出功率 [MW]
            info: 额外信息
        """
        action = int(action)
        
        # 解析动作
        if action == PSHAction.HOLD:
            pass  # 保持当前状态
            
        elif action == PSHAction.START_GENERATE:
            if self.current_mode == PSHMode.IDLE:
                if self.daily_cycle_count < self.max_daily_cycles and self._can_generate():
                    self.current_mode = PSHMode.GENERATING
                    self.operation_start_time = len(self.power_history) - 1
                    self.operation_duration = 0
                    self.daily_cycle_count += 1
                    
        elif action == PSHAction.START_PUMP:
            if self.current_mode == PSHMode.IDLE:
                if self.daily_cycle_count < self.max_daily_cycles and self._can_pump():
                    self.current_mode = PSHMode.PUMPING
                    self.operation_start_time = len(self.power_history) - 1
                    self.operation_duration = 0
                    self.daily_cycle_count += 1
                    
        elif action == PSHAction.STOP:
            if self.current_mode != PSHMode.IDLE:
                if self._check_duration_constraint():
                    self.current_mode = PSHMode.IDLE
                    self.operation_duration = 0
                    
        else:
            raise ValueError(f"Invalid PSH action: {action}")
        
        # 根据当前模式确定功率
        if self.current_mode == PSHMode.GENERATING:
            target_power = self.rated_gen_power
            energy_required = target_power * self.time_step / self.gen_efficiency
            if self.upper_energy - energy_required < self.upper_min:
                target_power = 0.0
                self.current_mode = PSHMode.IDLE
                self.operation_duration = 0
            else:
                self.operation_duration += 1
                
        elif self.current_mode == PSHMode.PUMPING:
            target_power = -self.rated_pump_power
            energy_stored = abs(target_power) * self.time_step * self.pump_efficiency
            if self.lower_energy + energy_stored > self.lower_capacity:
                target_power = 0.0
                self.current_mode = PSHMode.IDLE
                self.operation_duration = 0
            else:
                self.operation_duration += 1
        else:
            target_power = 0.0
            
        # 更新水库状态
        if self.current_mode == PSHMode.GENERATING:
            energy_converted = target_power * self.time_step / self.gen_efficiency
            self.upper_energy -= energy_converted
            self.lower_energy += energy_converted * self.gen_efficiency
            
        elif self.current_mode == PSHMode.PUMPING:
            energy_consumed = abs(target_power) * self.time_step
            energy_stored = energy_consumed * self.pump_efficiency
            self.lower_energy -= energy_consumed
            self.upper_energy += energy_stored
            
        # 确保水库状态在边界内
        self.upper_energy = max(self.upper_min, min(self.upper_energy, self.upper_capacity))
        self.lower_energy = max(self.lower_min, min(self.lower_energy, self.lower_capacity))
        
        # 更新状态
        self.current_power = target_power
        
        # 记录历史
        self.power_history.append(self.current_power)
        self.upper_soc_history.append(self.upper_soc)
        self.lower_soc_history.append(self.lower_soc)
        self.mode_history.append(self.current_mode)
        
        info = {
            'mode': self.current_mode,
            'upper_soc': self.upper_soc,
            'lower_soc': self.lower_soc,
            'operation_duration': self.operation_duration,
            'daily_cycles': self.daily_cycle_count,
            'action': action,
            'is_constraint_violated': False
        }
        
        return self.current_power, info
        
    def get_state(self) -> np.ndarray:
        """获取机组当前状态"""
        return np.array([
            self.upper_soc,
            self.lower_soc,
            self.current_power / self.rated_gen_power if self.current_power > 0 
            else self.current_power / self.rated_pump_power,
            float(self.current_mode),
            min(self.operation_duration / self.max_duration, 1.0),
            self.daily_cycle_count / self.max_daily_cycles
        ], dtype=np.float32)
        
    def get_constraints(self) -> Dict:
        """获取机组约束信息"""
        return {
            'rated_gen_power': self.rated_gen_power,
            'rated_pump_power': self.rated_pump_power,
            'upper_capacity': self.upper_capacity,
            'lower_capacity': self.lower_capacity,
            'max_daily_cycles': self.max_daily_cycles,
            'min_duration': self.min_duration,
            'max_duration': self.max_duration
        }


class BatteryEnergyStorageSystem:
    """电池储能系统(BESS)模型 - 15分钟快速调节"""
    
    def __init__(
        self,
        unit_id: int,
        node_id: int,
        max_power: float,
        capacity: float,
        min_soc: float,
        max_soc: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        initial_soc: float,
        ramp_rate_limit: float,
        time_step: float = 0.25
    ):
        self.unit_id = unit_id
        self.node_id = node_id
        self.max_power = max_power
        self.capacity = capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charge_eff = charge_efficiency
        self.discharge_eff = discharge_efficiency
        self.ramp_limit = ramp_rate_limit
        self.time_step = time_step
        
        self.current_soc = initial_soc
        self.current_power = 0.0
        
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        
    def reset(self, initial_soc: Optional[float] = None):
        """重置储能状态"""
        if initial_soc is not None:
            self.current_soc = initial_soc
        self.current_power = 0.0
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        
    def step(self, action: float) -> Tuple[float, float, Dict]:
        """执行一个时间步的动作 - 连续功率调节"""
        if action > 0:
            target_power = action * self.max_power
            target_power = min(target_power, self.max_power)
        elif action < 0:
            target_power = action * self.max_power
            target_power = max(target_power, -self.max_power)
        else:
            target_power = 0.0
            
        # 应用爬坡约束
        power_change = target_power - self.current_power
        if abs(power_change) > self.ramp_limit:
            power_change = np.sign(power_change) * self.ramp_limit
            target_power = self.current_power + power_change
            
        # 检查SOC约束
        if target_power > 0:
            energy_discharged = target_power * self.time_step / self.discharge_eff
            if self.current_soc * self.capacity - energy_discharged < self.min_soc * self.capacity:
                available_energy = (self.current_soc - self.min_soc) * self.capacity
                target_power = available_energy * self.discharge_eff / self.time_step
                target_power = max(0, target_power)
                
        elif target_power < 0:
            energy_charged = abs(target_power) * self.time_step * self.charge_eff
            if self.current_soc * self.capacity + energy_charged > self.max_soc * self.capacity:
                available_capacity = (self.max_soc - self.current_soc) * self.capacity
                target_power = -available_capacity / (self.charge_eff * self.time_step)
                target_power = min(0, target_power)
        
        # 更新SOC
        if target_power > 0:
            energy_change = -target_power * self.time_step / self.discharge_eff
        elif target_power < 0:
            energy_change = abs(target_power) * self.time_step * self.charge_eff
        else:
            energy_change = 0.0
            
        next_soc = self.current_soc + energy_change / self.capacity
        next_soc = max(self.min_soc, min(next_soc, self.max_soc))
        
        self.current_power = target_power
        self.current_soc = next_soc
        
        self.power_history.append(self.current_power)
        self.soc_history.append(self.current_soc)
        
        info = {
            'energy_change_mwh': energy_change,
            'is_constraint_violated': False
        }
        
        return self.current_power, next_soc, info
    
    def get_state(self) -> np.ndarray:
        """获取储能当前状态"""
        return np.array([
            self.current_soc,
            self.current_power / self.max_power
        ], dtype=np.float32)
    
    def get_constraints(self) -> Dict:
        """获取储能约束信息"""
        return {
            'max_power': self.max_power,
            'min_soc': self.min_soc,
            'max_soc': self.max_soc,
            'capacity': self.capacity,
            'ramp_limit': self.ramp_limit
        }
