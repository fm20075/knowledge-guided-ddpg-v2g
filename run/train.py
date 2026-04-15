"""
train.py
训练与测试主程序，增加记录风险阈值和最终SOC
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *
from core.v2g_env import V2GEnv
from core.ddpg import DDPG

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)

def main():
    env = V2GEnv()
    agent = DDPG(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_bound=ACTION_BOUND,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=LR_ACTOR,
        critic_lr=LR_CRITIC,
        memory_capacity=MEMORY_CAPACITY
    )

    training_log = []
    episode_reward_list = []
    risk_threshold_log = []

    print(f"开始训练，总回合数：{MAX_EPISODES}")
    for episode in tqdm(range(MAX_EPISODES)):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            if len(agent.memory) > BATCH_SIZE:
                critic_loss, actor_loss = agent.train(BATCH_SIZE)
            state = next_state
            episode_reward += reward

        episode_reward_list.append(episode_reward)
        training_log.append([episode + 1, episode_reward])

        if (episode + 1) % 50 == 0:
            threshold = max(0.03, 0.05 - (episode // 50) * 0.005)
            risk_threshold_log.append([episode + 1, threshold])
            print(f"回合 {episode+1}/{MAX_EPISODES}，总奖励：{episode_reward:.2f}，风险阈值：{threshold:.4f}")

    agent.save(MODEL_SAVE_PATH)
    print(f"模型已保存至 {MODEL_SAVE_PATH}")

    log_df = pd.DataFrame(training_log, columns=["episode", "total_reward"])
    log_df.to_csv(os.path.join(OUTPUT_SAVE_PATH, "training_log.csv"), index=False)
    print(f"训练日志已保存至 {OUTPUT_SAVE_PATH}")

    if risk_threshold_log:
        risk_df = pd.DataFrame(risk_threshold_log, columns=["episode", "risk_threshold"])
        risk_df.to_csv(os.path.join(OUTPUT_SAVE_PATH, "risk_threshold.csv"), index=False)
        print(f"风险阈值日志已保存至 {OUTPUT_SAVE_PATH}")

    # 测试阶段
    print("开始测试模型，复现论文结果")
    state = env.reset()
    done = False
    test_metrics = []
    final_soc_records = []

    while not done:
        action = agent.select_action(state, add_noise=False)
        next_state, reward, done, info = env.step(action)
        test_metrics.append([
            env.current_time_step,
            info["current_risk"],
            info["load_rate"],
            info["ev_net_power"],
            info["avg_soc"],
            reward
        ])
        state = next_state

    for ev in env.ev_aggregator.ev_data:
        final_soc_records.append({
            "ev_id": ev["ev_id"],
            "access_time": ev["access_time"],
            "leave_time": ev["leave_time"],
            "init_soc": ev["init_soc"],
            "target_soc": ev["target_soc"],
            "final_soc": ev["current_soc"]
        })

    test_metrics_df = pd.DataFrame(test_metrics, columns=[
        "time_slot", "risk_prob", "load_rate", "ev_net_power", "avg_soc", "step_reward"
    ])
    test_metrics_df.to_csv(os.path.join(OUTPUT_SAVE_PATH, "final_test_metrics.csv"), index=False)

    final_soc_df = pd.DataFrame(final_soc_records)
    final_soc_df.to_csv(os.path.join(OUTPUT_SAVE_PATH, "final_soc.csv"), index=False)
    print(f"EV最终SOC已保存至 {OUTPUT_SAVE_PATH}")

    final_avg_risk = np.mean(test_metrics_df["risk_prob"])
    final_max_risk = np.max(test_metrics_df["risk_prob"])
    final_avg_load_rate = np.mean(test_metrics_df["load_rate"])
    final_total_reward = np.sum(test_metrics_df["step_reward"])

    with open(os.path.join(OUTPUT_SAVE_PATH, "final_test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("论文核心实验指标结果\n")
        f.write("========================\n")
        f.write(f"平均联合风险概率：{final_avg_risk:.4f}\n")
        f.write(f"最大风险概率：{final_max_risk:.4f}\n")
        f.write(f"平均电网负荷率：{final_avg_load_rate:.4f}\n")
        f.write(f"测试总奖励：{final_total_reward:.2f}\n")

    print("\n===== 论文核心实验指标 =====")
    print(f"平均联合风险概率：{final_avg_risk:.4f}")
    print(f"最大风险概率：{final_max_risk:.4f}")
    print(f"平均电网负荷率：{final_avg_load_rate:.4f}")
    print(f"测试总奖励：{final_total_reward:.2f}")

if __name__ == "__main__":
    main()