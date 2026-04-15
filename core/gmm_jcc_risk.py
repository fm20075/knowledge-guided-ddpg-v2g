"""
gmm_jcc_risk.py
GMM-JCC联合风险概率估计模块
对应论文章节：4 联合风险概率评估模型（核心创新章节）
所有函数均标注对应论文的公式编号
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from core.config import *

class GMMJCCRiskEstimator:
    def __init__(self):
        self.gmm_model = GaussianMixture(n_components=GMM_COMPONENTS, random_state=42)
        self.is_trained = False

    def fit_risk_model(self, historical_data):
        """
        拟合GMM风险模型，对应论文公式(4-2)
        historical_data: numpy array, 形状 (n_samples, 3)
                        每行 [grid_load, ev_power, voltage_deviation]
        """
        self.gmm_model.fit(historical_data)
        self.is_trained = True

    def calculate_joint_risk(self, grid_load, ev_power, voltage_deviation):
        """
        计算联合风险概率，对应论文公式(4-7)（核心创新公式）
        返回：风险概率值 (0~1)
        """
        # 如果模型未训练，使用简化经验公式（备用方案）
        if not self.is_trained:
            load_rate = grid_load / GRID_CAPACITY
            voltage_risk = abs(voltage_deviation) / VOLTAGE_DEVIATION_MAX
            power_risk = abs(ev_power) / (EV_NUM * EV_CHARGING_POWER_MAX)
            risk_prob = 0.6 * load_rate + 0.25 * voltage_risk + 0.15 * power_risk
            return np.clip(risk_prob, 0, 1)

        # 使用 GMM 计算联合风险概率
        feature_vector = np.array([[grid_load, ev_power, voltage_deviation]])
        # 计算样本在 GMM 下的似然密度
        densities = np.exp(self.gmm_model.score_samples(feature_vector))
        # 取 GMM 各高斯中心的最大密度作为参考
        max_density = np.max(np.exp(self.gmm_model.score_samples(self.gmm_model.means_)))
        # 风险概率定义为 1 - 相对密度（密度越低风险越高）
        risk_prob = 1 - np.clip(densities[0] / max_density, 0, 1)
        return risk_prob