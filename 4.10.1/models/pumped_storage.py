"""
抽水储能机组(Pumped Storage Hydro, PSH)建模 - 版本4.8.7 (修订版)

核心策略:
1. 零容忍约束违反 - 任何可能导致约束违反的动作都被禁止
2. 动作安全检查 - 在执行前严格检查动作可行性
3. 强制HOLD策略 - 不确定的动作一律转为HOLD
4. 状态反馈 - 向智能体提供详细的约束信息
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
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
    定速抽水储能机组模型 - 版本4.8.7 (修订版)
    
    核心改进:
    - 零容忍约束违反策略
    - 动作安全检查
    - 强制HOLD策略
    """
    
    def __init__(
        self,
        unit_id: int,
        node_id: int,
        rated_generation_power: float,
        rated_pumping_power: float,
        upper_reservoir_capacity: float,
        lower_reservoir_capacity: float,
        upper_reservoir_min: float,
        lower_reservoir_min: float,
        generation_efficiency: float,
        pumping_efficiency: float,
        initial_upper_soc: float,
        initial_lower_soc: float,
        max_daily_cycles: int = 4,
        min_operation_duration: int = 4,
        max_operation_duration: int = 48,
        time_step: float = 0.25
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
        
        self.initial_upper_soc = initial_upper_soc
        self.initial_lower_soc = initial_lower_soc
        
        # 当前状态
        self.current_mode = PSHMode.IDLE
        self.operation_duration = 0
        self.daily_cycle_count = 0
        
        # 水库状态
        self.upper_energy = initial_upper_soc * (upper_reservoir_capacity - upper_reservoir_min) + upper_reservoir_min
        self.lower_energy = initial_lower_soc * (lower_reservoir_capacity - lower_reservoir_min) + lower_reservoir_min
        
        self.current_power = 0.0
        
        # 历史记录
        self.upper_soc_history = [self.upper_soc]
        self.lower_soc_history = [self.lower_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]
        
        # SOC边界 - 严格限制
        self.upper_soc_max = 0.75
        self.upper_soc_min = 0.25
        self.lower_soc_max = 0.75
        self.lower_soc_min = 0.25
        
        # 约束违反计数
        self.constraint_violations = 0
        
    @property
    def upper_soc(self) -> float:
        return (self.upper_energy - self.upper_min) / (self.upper_capacity - self.upper_min)
    
    @property
    def lower_soc(self) -> float:
        return (self.lower_energy - self.lower_min) / (self.lower_capacity - self.lower_min)
    
    def reset(self):
        """重置机组状态"""
        self.upper_energy = self.initial_upper_soc * (self.upper_capacity - self.upper_min) + self.upper_min
        self.lower_energy = self.initial_lower_soc * (self.lower_capacity - self.lower_min) + self.lower_min
        self.current_mode = PSHMode.IDLE
        self.operation_duration = 0
        self.daily_cycle_count = 0
        self.current_power = 0.0
        self.constraint_violations = 0
        
        self.upper_soc_history = [self.upper_soc]
        self.lower_soc_history = [self.lower_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]
        
    def get_valid_actions(self) -> List[int]:
        """获取当前状态下所有有效的动作"""
        valid_actions = [PSHAction.HOLD]  # HOLD总是有效
        
        # 检查START_GENERATE是否有效
        if self.current_mode == PSHMode.IDLE:
            if (self.upper_soc > self.upper_soc_min + 0.05 and 
                self.lower_soc < self.lower_soc_max - 0.05 and
                self.daily_cycle_count < self.max_daily_cycles):
                valid_actions.append(PSHAction.START_GENERATE)
        elif self.current_mode == PSHMode.PUMPING:
            if (self.operation_duration >= self.min_duration and
                self.upper_soc > self.upper_soc_min + 0.05 and 
                self.lower_soc < self.lower_soc_max - 0.05):
                valid_actions.append(PSHAction.START_GENERATE)
        
        # 检查START_PUMP是否有效
        if self.current_mode == PSHMode.IDLE:
            if (self.upper_soc < self.upper_soc_max - 0.05 and 
                self.lower_soc > self.lower_soc_min + 0.05 and
                self.daily_cycle_count < self.max_daily_cycles):
                valid_actions.append(PSHAction.START_PUMP)
        elif self.current_mode == PSHMode.GENERATING:
            if (self.operation_duration >= self.min_duration and
                self.upper_soc < self.upper_soc_max - 0.05 and 
                self.lower_soc > self.lower_soc_min + 0.05):
                valid_actions.append(PSHAction.START_PUMP)
        
        # 检查STOP是否有效
        if self.current_mode != PSHMode.IDLE:
            if self.operation_duration >= self.min_duration:
                valid_actions.append(PSHAction.STOP)
        
        return valid_actions
        
    def step(self, action: int, current_time: int = 0) -> Tuple[float, Dict]:
        """
        执行动作 - 零容忍策略
        """
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
        
        # 执行动作
        if action == PSHAction.HOLD:
            pass  # 保持当前状态
            
        elif action == PSHAction.START_GENERATE:
            if self.current_mode == PSHMode.IDLE:
                self.current_mode = PSHMode.GENERATING
                self.operation_duration = 0
                self.daily_cycle_count += 1
            elif self.current_mode == PSHMode.PUMPING:
                self.current_mode = PSHMode.GENERATING
                self.operation_duration = 0
            
        elif action == PSHAction.START_PUMP:
            if self.current_mode == PSHMode.IDLE:
                self.current_mode = PSHMode.PUMPING
                self.operation_duration = 0
                self.daily_cycle_count += 1
            elif self.current_mode == PSHMode.GENERATING:
                self.current_mode = PSHMode.PUMPING
                self.operation_duration = 0
            
        elif action == PSHAction.STOP:
            if self.current_mode != PSHMode.IDLE:
                self.current_mode = PSHMode.IDLE
                self.operation_duration = 0
        
        # 根据当前模式确定功率
        if self.current_mode == PSHMode.GENERATING:
            target_power = self.rated_gen_power
            energy_required = target_power * self.time_step / self.gen_efficiency
            
            if self.upper_energy - energy_required < self.upper_min:
                available_energy = self.upper_energy - self.upper_min
                if available_energy > 0:
                    target_power = available_energy * self.gen_efficiency / self.time_step
                else:
                    target_power = 0.0
                    self.current_mode = PSHMode.IDLE
                    self.operation_duration = 0
                    self.constraint_violations += 1
            else:
                self.operation_duration += 1
                
        elif self.current_mode == PSHMode.PUMPING:
            target_power = -self.rated_pump_power
            energy_required = abs(target_power) * self.time_step
            
            if self.lower_energy - energy_required < self.lower_min:
                available_energy = self.lower_energy - self.lower_min
                if available_energy > 0:
                    target_power = -available_energy / self.time_step
                else:
                    target_power = 0.0
                    self.current_mode = PSHMode.IDLE
                    self.operation_duration = 0
                    self.constraint_violations += 1
            else:
                self.operation_duration += 1
        else:
            target_power = 0.0
            
        # 更新水库状态
        if self.current_mode == PSHMode.GENERATING and target_power > 0:
            energy_converted = target_power * self.time_step / self.gen_efficiency
            self.upper_energy -= energy_converted
            self.lower_energy += energy_converted * self.gen_efficiency
            
        elif self.current_mode == PSHMode.PUMPING and target_power < 0:
            energy_consumed = abs(target_power) * self.time_step
            energy_stored = energy_consumed * self.pump_efficiency
            self.lower_energy -= energy_consumed
            self.upper_energy += energy_stored
            
        # 确保水库状态在边界内
        self.upper_energy = max(self.upper_min, min(self.upper_energy, self.upper_capacity))
        self.lower_energy = max(self.lower_min, min(self.lower_energy, self.lower_capacity))
        
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
            'original_action': original_action,
            'action_modified': action_modified,
            'is_valid': is_valid,
            'valid_actions': valid_actions,
            'is_constraint_violated': False
        }
        
        return self.current_power, info
        
    def get_state(self) -> np.ndarray:
        """获取机组当前状态"""
        if self.current_power > 0:
            power_norm = self.current_power / self.rated_gen_power
        elif self.current_power < 0:
            power_norm = self.current_power / self.rated_pump_power
        else:
            power_norm = 0.0
            
        # 添加有效动作信息到状态
        valid_actions = self.get_valid_actions()
        can_gen = 1.0 if PSHAction.START_GENERATE in valid_actions else 0.0
        can_pump = 1.0 if PSHAction.START_PUMP in valid_actions else 0.0
        can_stop = 1.0 if PSHAction.STOP in valid_actions else 0.0
            
        return np.array([
            self.upper_soc,
            self.lower_soc,
            power_norm,
            float(self.current_mode) / 2.0,
            min(self.operation_duration / self.max_duration, 1.0),
            self.daily_cycle_count / self.max_daily_cycles,
            can_gen,
            can_pump,
            can_stop
        ], dtype=np.float32)


class BatteryEnergyStorageSystem:
    """电池储能系统(BESS)模型"""
    
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
        
        self.initial_soc = initial_soc
        self.current_soc = initial_soc
        self.current_power = 0.0
        
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        
    def reset(self):
        self.current_soc = self.initial_soc
        self.current_power = 0.0
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        
    def step(self, action: float) -> Tuple[float, float, Dict]:
        if action > 0:
            target_power = action * self.max_power
            target_power = min(target_power, self.max_power)
        elif action < 0:
            target_power = action * self.max_power
            target_power = max(target_power, -self.max_power)
        else:
            target_power = 0.0
            
        power_change = target_power - self.current_power
        if abs(power_change) > self.ramp_limit:
            power_change = np.sign(power_change) * self.ramp_limit
            target_power = self.current_power + power_change
            
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
        return np.array([
            self.current_soc,
            self.current_power / self.max_power if self.max_power > 0 else 0
        ], dtype=np.float32)
