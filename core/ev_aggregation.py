"""
ev_aggregation.py
PPIPAM电动汽车聚合模块
对应论文章节：3.2 电动汽车聚合控制策略
"""
import numpy as np
from core.config import *

class EVAggregator:
    def __init__(self, ev_data):
        """
        初始化聚合器
        ev_data: list of dict, 每个元素包含 ev_id, access_time, leave_time, 
                 init_soc, target_soc, current_soc
        """
        self.ev_data = ev_data
        self.ev_num = len(ev_data)

    def calculate_adjustable_power(self, current_time):
        """
        计算当前时段 EV 集群的总可调功率
        对应论文公式(3-5)
        返回:
            total_chargable_power: 可充电总功率 (kW)
            total_dischargable_power: 可放电总功率 (kW)
            avg_soc: 当前在线 EV 的平均 SOC
            current_ev_soc_list: 当前在线 EV 的 SOC 列表
        """
        total_chargable_power = 0.0
        total_dischargable_power = 0.0
        current_ev_soc_list = []

        for ev in self.ev_data:
            if current_time >= ev["access_time"] and current_time < ev["leave_time"]:
                current_soc = ev["current_soc"]
                current_ev_soc_list.append(current_soc)
                if current_soc < EV_SOC_MAX:
                    total_chargable_power += EV_CHARGING_POWER_MAX
                if current_soc > EV_SOC_MIN:
                    total_dischargable_power += EV_DISCHARGING_POWER_MAX

        avg_soc = np.mean(current_ev_soc_list) if len(current_ev_soc_list) > 0 else 0.5
        return total_chargable_power, total_dischargable_power, avg_soc, current_ev_soc_list

    def execute_dispatch(self, action, current_time):
        """
        执行充放电调度指令，更新 EV SOC
        对应论文公式(3-8)
        action: [total_charge_power_ratio, total_discharge_power_ratio, charge_coeff, discharge_coeff]
               但此处我们将 action 直接理解为 4 维连续值:
               action[0] = 总充电功率 (kW)，已在 [-max, max] 范围内
               action[1] = 总放电功率 (kW)
               action[2] = 充电分配系数 (0~1)
               action[3] = 放电分配系数 (0~1)
        返回:
            ev_net_power: 净充放电功率 (充电为正，放电为负)
        """
        # 从动作中提取指令（注意：action 已由 DDPG 输出并 clip 到合理范围）
        total_charge_power = np.clip(action[0], 0, EV_CHARGING_POWER_MAX * self.ev_num)
        total_discharge_power = np.clip(action[1], 0, EV_DISCHARGING_POWER_MAX * self.ev_num)
        charge_coeff = np.clip(action[2], 0, 1)
        discharge_coeff = np.clip(action[3], 0, 1)

        # 筛选当前在线且可充/可放的 EV
        charge_ev_list = []
        discharge_ev_list = []
        for ev in self.ev_data:
            if current_time >= ev["access_time"] and current_time < ev["leave_time"]:
                if ev["current_soc"] < EV_SOC_MAX:
                    charge_ev_list.append(ev)
                if ev["current_soc"] > EV_SOC_MIN:
                    discharge_ev_list.append(ev)

        # 充电分配：按 SOC 升序，优先给电量低的充电
        if len(charge_ev_list) > 0 and total_charge_power > 0:
            charge_ev_list.sort(key=lambda x: x["current_soc"])  # 低 SOC 优先
            # 分配功率：系数决定是更偏向平均分配 (charge_coeff=0) 还是更偏向优先级 (charge_coeff=1)
            for i, ev in enumerate(charge_ev_list):
                # 功率分配比例：基础比例 + 优先级加成
                power_ratio = (1 - charge_coeff) * (i / len(charge_ev_list)) + charge_coeff
                ev_charge_power = total_charge_power * power_ratio / len(charge_ev_list)
                ev_charge_power = min(ev_charge_power, EV_CHARGING_POWER_MAX)
                soc_increase = (ev_charge_power * TIME_INTERVAL) / EV_BATTERY_CAPACITY
                ev["current_soc"] = min(ev["current_soc"] + soc_increase, EV_SOC_MAX)

        # 放电分配：按 SOC 降序，优先给电量高的放电
        if len(discharge_ev_list) > 0 and total_discharge_power > 0:
            discharge_ev_list.sort(key=lambda x: -x["current_soc"])  # 高 SOC 优先
            for i, ev in enumerate(discharge_ev_list):
                power_ratio = (1 - discharge_coeff) * (i / len(discharge_ev_list)) + discharge_coeff
                ev_discharge_power = total_discharge_power * power_ratio / len(discharge_ev_list)
                ev_discharge_power = min(ev_discharge_power, EV_DISCHARGING_POWER_MAX)
                soc_decrease = (ev_discharge_power * TIME_INTERVAL) / EV_BATTERY_CAPACITY
                ev["current_soc"] = max(ev["current_soc"] - soc_decrease, EV_SOC_MIN)

        # 净功率（充电为正，放电为负）
        ev_net_power = total_charge_power - total_discharge_power
        return ev_net_power