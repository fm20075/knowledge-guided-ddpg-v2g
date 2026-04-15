"""
generate_simulation_data.py
杭州西湖区停车场V2G仿真数据生成
贴合工程场景：西湖区商业停车场，工作日早8点-晚10点EV接入高峰
"""
import numpy as np
import os
import sys

# 添加项目根目录到路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *

# 设置随机种子以保证可复现性
np.random.seed(42)

# 时间点：0~23 小时
time_slots = np.arange(TIME_SLOT)

# 电网基础负荷曲线（kW），模拟杭州西湖区典型日负荷
grid_base_load = np.array([
    320, 300, 280, 270, 260, 280, 350, 500, 750, 820, 850, 830,
    800, 780, 820, 860, 900, 880, 750, 600, 450, 400, 360, 330
], dtype=np.float32)

# 生成 EV 数据
ev_data = []
for ev_id in range(EV_NUM):
    # 接入时间：8~22 点之间随机
    access_time = np.random.randint(8, 22)
    # 停留时长：2~8 小时
    stay_duration = np.random.randint(2, 9)
    # 离开时间不能超过 23 点
    leave_time = min(access_time + stay_duration, 23)
    # 初始 SOC：0.2~0.6 之间均匀分布
    init_soc = np.random.uniform(EV_SOC_MIN, 0.6)
    # 目标 SOC：0.8~0.95 之间均匀分布
    target_soc = np.random.uniform(0.8, EV_SOC_MAX)
    ev_data.append({
        "ev_id": ev_id,
        "access_time": access_time,
        "leave_time": leave_time,
        "init_soc": init_soc,
        "target_soc": target_soc,
        "current_soc": init_soc  # 初始 SOC 即为当前 SOC
    })

# 组装数据集
simulation_dataset = {
    "time_slots": time_slots,
    "grid_base_load": grid_base_load,
    "ev_data": ev_data
}

# 保存为 .npy 文件
os.makedirs("./data", exist_ok=True)
np.save("./data/simulation_data.npy", simulation_dataset, allow_pickle=True)

print("仿真数据已生成，保存至 ./data/simulation_data.npy")
print(f"EV 数量：{EV_NUM}")
print(f"时段数：{TIME_SLOT}")
print("电网基础负荷曲线（kW）：", grid_base_load)