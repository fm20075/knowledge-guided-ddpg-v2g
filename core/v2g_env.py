"""
v2g_env.py
V2G协同保护仿真环境
对应论文章节：3 系统模型与问题描述
状态空间：11维，与论文完全一致
动作空间：4维，与论文完全一致
"""
import numpy as np
from core.config import *
from core.gmm_jcc_risk import GMMJCCRiskEstimator
from core.ev_aggregation import EVAggregator

class V2GEnv:
    def __init__(self):
        # 加载仿真数据
        self.simulation_data = np.load(DATA_PATH, allow_pickle=True).item()
        self.time_slots = self.simulation_data["time_slots"]
        self.grid_base_load = self.simulation_data["grid_base_load"]
        self.ev_origin_data = self.simulation_data["ev_data"]

        # 初始化风险估计器
        self.risk_estimator = GMMJCCRiskEstimator()
        self._pretrain_risk_model()  # 用模拟历史数据预训练

        # 环境状态
        self.current_time_step = 0
        self.ev_aggregator = None
        self.last_ev_power = 0.0
        self.current_state = None
        self.reset()

    def _pretrain_risk_model(self):
        """用模拟历史数据预训练GMM风险模型"""
        np.random.seed(42)
        historical_data = []
        for _ in range(1000):
            random_load = np.random.normal(np.mean(self.grid_base_load), np.std(self.grid_base_load))
            random_ev_power = np.random.normal(0, EV_NUM * EV_CHARGING_POWER_MAX * 0.3)
            random_voltage_dev = np.random.normal(0, 0.03)
            historical_data.append([random_load, random_ev_power, random_voltage_dev])
        self.risk_estimator.fit_risk_model(np.array(historical_data))

    def _get_state(self):
        """获取当前时刻的11维状态，对应论文公式(5-1)"""
        total_chargable, total_dischargable, avg_soc, _ = self.ev_aggregator.calculate_adjustable_power(self.current_time_step)
        current_grid_load = self.grid_base_load[self.current_time_step] + self.last_ev_power
        load_rate = current_grid_load / GRID_CAPACITY
        voltage_deviation = (load_rate - 1) * 0.08  # 简化电压偏差模型
        current_risk = self.risk_estimator.calculate_joint_risk(current_grid_load, self.last_ev_power, voltage_deviation)
        online_ev_num = len([ev for ev in self.ev_aggregator.ev_data if self.current_time_step >= ev["access_time"] and self.current_time_step < ev["leave_time"]])
        remaining_time = TIME_SLOT - self.current_time_step - 1

        state = np.array([
            self.current_time_step,          # 当前时段
            current_grid_load,               # 当前总负荷
            load_rate,                       # 负荷率
            total_chargable,                 # 可充电总功率
            total_dischargable,              # 可放电总功率
            avg_soc,                         # 平均SOC
            current_risk,                    # 当前联合风险概率
            voltage_deviation,               # 电压偏差
            self.last_ev_power,              # 上一时段EV净功率
            remaining_time,                  # 剩余时段数
            online_ev_num                    # 在线EV数量
        ], dtype=np.float32)
        return state

    def step(self, action):
        """
        环境步进函数，执行动作并返回新状态、奖励、结束标志
        对应论文公式(5-3) 奖励函数定义
        """
        # 执行调度，更新EV SOC
        ev_net_power = self.ev_aggregator.execute_dispatch(action, self.current_time_step)
        self.last_ev_power = ev_net_power

        # 计算当前指标
        current_grid_load = self.grid_base_load[self.current_time_step] + ev_net_power
        load_rate = current_grid_load / GRID_CAPACITY
        voltage_deviation = (load_rate - 1) * 0.08
        current_risk = self.risk_estimator.calculate_joint_risk(current_grid_load, ev_net_power, voltage_deviation)

        # 奖励函数
        # 风险惩罚：超过阈值则严重惩罚，否则给予正奖励
        if current_risk > RISK_THRESHOLD:
            risk_penalty = -10 * current_risk
        else:
            risk_penalty = 2 * (1 - current_risk)
        # 负荷成本：偏离目标负荷率0.7的惩罚
        load_cost = -abs(load_rate - 0.7) * 5
        # 用户满意度：对于即将离开的EV，根据SOC偏离目标的程度给予奖励
        satisfaction_reward = 0
        for ev in self.ev_aggregator.ev_data:
            if self.current_time_step == ev["leave_time"] - 1:  # 最后一小时
                soc_deviation = abs(ev["current_soc"] - ev["target_soc"])
                satisfaction_reward += 3 * (1 - soc_deviation)
        reward = risk_penalty + load_cost + satisfaction_reward

        # 步进时间
        self.current_time_step += 1
        done = self.current_time_step >= TIME_SLOT
        if not done:
            self.current_state = self._get_state()
        else:
            self.current_state = np.zeros(STATE_DIM)

        return self.current_state, reward, done, {
            "current_risk": current_risk,
            "load_rate": load_rate,
            "ev_net_power": ev_net_power,
            "avg_soc": self.current_state[5] if not done else 0.0
        }

    def reset(self):
        """重置环境，开始新的回合"""
        self.current_time_step = 0
        self.last_ev_power = 0.0
        # 深拷贝原始EV数据，避免相互影响
        reset_ev_data = [ev.copy() for ev in self.ev_origin_data]
        self.ev_aggregator = EVAggregator(reset_ev_data)
        self.current_state = self._get_state()
        return self.current_state