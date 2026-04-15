"""
config.py
全局参数配置文件，对齐论文符号
"""
# ===================== 系统基础参数 =====================
EV_NUM = 50               # N_{EV}：电动汽车数量
TIME_SLOT = 24            # T：调度时段数
GRID_CAPACITY = 1000      # P_{grid_max}：配电网最大容量（kW）
TIME_INTERVAL = 1         # 时段时长（小时）

# ===================== 风险评估参数 =====================
GMM_COMPONENTS = 3        # K：GMM高斯分量数
RISK_THRESHOLD = 0.85     # λ_{risk}：风险预警阈值
VOLTAGE_NOMINAL = 220     # 额定电压（V）
VOLTAGE_DEVIATION_MAX = 0.1  # 最大允许电压偏差

# ===================== EV聚合参数 =====================
EV_CHARGING_POWER_MAX = 7     # P_{ch_max}（kW）
EV_DISCHARGING_POWER_MAX = 7  # P_{dis_max}（kW）
EV_SOC_MIN = 0.2              # SOC_min
EV_SOC_MAX = 0.95             # SOC_max
EV_BATTERY_CAPACITY = 70      # 电池容量（kWh）

# ===================== DDPG算法超参数 =====================
STATE_DIM = 11                # 状态空间维度
ACTION_DIM = 4                # 动作空间维度
ACTION_BOUND = 1.0            # 动作范围
MAX_EPISODES = 500            # 最大训练回合数
LR_ACTOR = 0.0001             # Actor学习率
LR_CRITIC = 0.0002            # Critic学习率
GAMMA = 0.99                  # 折扣因子
TAU = 0.005                   # 软更新系数
MEMORY_CAPACITY = 100000      # 经验池容量
BATCH_SIZE = 256              # 批量大小

# ===================== 文件路径 =====================
DATA_PATH = "./data/simulation_data.npy"
MODEL_SAVE_PATH = "./model/"
OUTPUT_SAVE_PATH = "./output/"
FIGURE_SAVE_PATH = "./output/figures/"