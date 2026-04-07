"""
数据生成工具
生成IEEE 34节点配电网的负荷、光伏和电价数据
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_ieee34_node_data(
        n_nodes: int = 34,
        n_time_steps: int = 35040,
        time_step: float = 0.25,
        random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成IEEE 34节点配电网数据"""
    np.random.seed(random_seed)

    time_index = pd.date_range(
        start='2024-01-01',
        periods=n_time_steps,
        freq='15min'
    )

    load_data = generate_load_data(n_nodes, n_time_steps, time_index)
    pv_data = generate_pv_data(n_nodes, n_time_steps, time_index)
    price_data = generate_price_data(n_time_steps, time_index)

    return load_data, pv_data, price_data


def generate_load_data(n_nodes: int, n_time_steps: int, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """生成负荷数据"""
    base_load_curve = np.array([
        0.65, 0.62, 0.60, 0.58, 0.57, 0.58, 0.62, 0.70,
        0.78, 0.85, 0.90, 0.92, 0.93, 0.92, 0.90, 0.88,
        0.87, 0.88, 0.92, 0.98, 1.00, 0.98, 0.95, 0.92,
        0.90, 0.88, 0.87, 0.85, 0.83, 0.82, 0.80, 0.78,
        0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70,
        0.70, 0.71, 0.73, 0.76, 0.80, 0.85, 0.90, 0.95,
        0.98, 1.00, 1.00, 0.98, 0.95, 0.92, 0.90, 0.88,
        0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80,
        0.80, 0.82, 0.85, 0.90, 0.95, 0.98, 1.00, 0.98,
        0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.74, 0.70,
        0.68, 0.66, 0.65, 0.64, 0.64, 0.65, 0.66, 0.67,
        0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.60
    ])

    node_base_loads = np.random.uniform(50, 500, n_nodes)
    load_data = np.zeros((n_time_steps, n_nodes))

    for t in range(n_time_steps):
        time_of_day = t % 96
        base_load = base_load_curve[time_of_day]

        day_of_week = (t // 96) % 7
        if day_of_week >= 5:
            base_load *= 0.85

        day_of_year = t // 96
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)

        for n in range(n_nodes):
            noise = np.random.normal(0, 0.05)
            load_data[t, n] = node_base_loads[n] * base_load * seasonal_factor * (1 + noise)

    load_df = pd.DataFrame(
        load_data,
        index=time_index,
        columns=[f'load_node_{i + 1}' for i in range(n_nodes)]
    )

    return load_df


def generate_pv_data(n_nodes: int, n_time_steps: int, time_index: pd.DatetimeIndex, 
                     pv_penetration: float = 0.3) -> pd.DataFrame:
    """生成光伏出力数据"""
    n_pv_nodes = int(n_nodes * pv_penetration)
    pv_nodes = np.random.choice(n_nodes, n_pv_nodes, replace=False)
    pv_capacities = np.random.uniform(100, 500, n_pv_nodes)

    pv_data = np.zeros((n_time_steps, n_nodes))

    for t in range(n_time_steps):
        hour = (t % 96) * 0.25

        if 6 <= hour <= 18:
            irradiance = np.sin(np.pi * (hour - 6) / 12)
            cloud_factor = np.random.uniform(0.7, 1.0)
            day_of_year = t // 96
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            irradiance = irradiance * cloud_factor * seasonal_factor
        else:
            irradiance = 0.0

        for i, node_idx in enumerate(pv_nodes):
            noise = np.random.normal(0, 0.02)
            pv_power = pv_capacities[i] * irradiance * max(0, 1 + noise)
            pv_data[t, node_idx] = pv_power

    pv_df = pd.DataFrame(
        pv_data,
        index=time_index,
        columns=[f'pv_node_{i + 1}' for i in range(n_nodes)]
    )

    return pv_df


def generate_price_data(n_time_steps: int, time_index: pd.DatetimeIndex) -> pd.DataFrame:
    """生成电价数据"""
    base_price_curve = np.array([
        35, 33, 32, 31, 30, 31, 33, 38,
        42, 45, 48, 50, 52, 51, 50, 48,
        46, 45, 48, 52, 55, 53, 50, 47,
        45, 43, 42, 40, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27,
        28, 30, 33, 37, 42, 48, 55, 62,
        68, 72, 75, 73, 70, 65, 60, 55,
        52, 50, 48, 46, 44, 42, 40, 38,
        37, 40, 45, 52, 60, 68, 75, 80,
        78, 72, 65, 58, 52, 48, 45, 42,
        40, 38, 36, 35, 34, 33, 32, 31,
        30, 29, 28, 27, 26, 26, 25, 25
    ])

    prices = np.zeros(n_time_steps)

    for t in range(n_time_steps):
        time_of_day = t % 96
        base_price = base_price_curve[time_of_day]
        noise = np.random.normal(0, 3)
        day_of_year = t // 96
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
        prices[t] = base_price * seasonal_factor + noise

    prices = np.maximum(prices, 10.0)

    price_df = pd.DataFrame({'price': prices}, index=time_index)

    return price_df


def create_ieee34_node_topology() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """创建IEEE 34节点配电网拓扑"""
    node_data = {
        'bus_id': list(range(1, 35)),
        'type': [2] + [0] * 33,
        'Pd': [0] * 34,
        'Qd': [0] * 34,
        'Gs': [0] * 34,
        'Bs': [0] * 34,
        'Vmax': [1.05] * 34,
        'Vmin': [0.95] * 34,
    }

    node_df = pd.DataFrame(node_data)

    line_data = {
        'from_bus': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                     31, 32, 33],
        'to_bus': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                   22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                   32, 33, 34],
        'r': [0.0922, 0.4930, 0.3660, 0.3811, 0.8190, 0.1872, 0.7114, 1.0300,
              1.0440, 0.1966, 0.3744, 1.4680, 0.5416, 0.5910, 0.7463, 1.2890,
              0.7320, 0.1640, 1.5042, 0.4095, 0.7089, 0.4512, 0.8980, 0.8960,
              0.8960, 0.8960, 0.8960, 0.8960, 0.8960, 0.8960, 0.8960, 0.8960, 0.8960],
        'x': [0.0470, 0.2511, 0.1864, 0.1941, 0.7070, 0.6188, 0.2351, 0.7400,
              0.7400, 0.0650, 0.1238, 0.3850, 0.2809, 0.3930, 0.2890, 0.1710,
              0.2511, 0.3709, 0.2511, 0.3709, 0.2511, 0.3709, 0.2511, 0.3709,
              0.3709, 0.3709, 0.3709, 0.3709, 0.3709, 0.3709, 0.3709, 0.3709, 0.3709],
        'b': [0.0] * 33,
        'rateA': [1000] * 33,
    }

    line_df = pd.DataFrame(line_data)

    return node_df, line_df


def save_data(load_data: pd.DataFrame, pv_data: pd.DataFrame, price_data: pd.DataFrame,
              node_data: pd.DataFrame, line_data: pd.DataFrame, output_dir: str):
    """保存数据到文件"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    load_data.to_csv(f"{output_dir}/load_data.csv")
    pv_data.to_csv(f"{output_dir}/pv_data.csv")
    price_data.to_csv(f"{output_dir}/price_data.csv")
    node_data.to_csv(f"{output_dir}/node_data.csv", index=False)
    line_data.to_csv(f"{output_dir}/line_data.csv", index=False)

    print(f"数据已保存到 {output_dir}")


if __name__ == "__main__":
    print("正在生成IEEE 34节点配电网数据...")
    load_data, pv_data, price_data = generate_ieee34_node_data()
    node_data, line_data = create_ieee34_node_topology()
    save_data(load_data, pv_data, price_data, node_data, line_data, 
              "/mnt/okcomputer/output/psh_scheme2/data")
    print("数据生成完成!")
