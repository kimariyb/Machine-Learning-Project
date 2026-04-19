import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from utils import data_processing
from loguru import logger
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 兼容Windows/Linux/Mac
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (12, 5)  # 设置画布整体大小

class PowerLoadPredictor:
    def __init__(self, data_path, model_path, target_col):
        self.target_col = target_col
        self.model_path = model_path
        self.split_time = pd.Timestamp("2015-08-01 00:00:00")
        self.X = None
        self.y = None
        os.makedirs(self.model_path, exist_ok=True)

        self._setup_logger()
        self._load_data()

        logger.success("PowerLoadPredictor 初始化完成")

    def _load_data(self):
        data_dir = "./data"
        self.test_path = os.path.join(data_dir, "test.csv")
        df = data_processing(self.test_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        self.history_data = df[df["time"] < self.split_time].copy()
        self.future_data = df[df["time"] >= self.split_time].copy()

        logger.info(f"历史数据: {self.history_data.shape}")
        logger.info(f"预测区间: {self.future_data.shape}")

    def _setup_logger(self):
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

        log_filename = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)

        logger.remove()

        logger.add(
            log_path,
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
            enqueue=True
        )

        logger.add(lambda msg: print(msg, end=""))

        self.log_path = log_path
        logger.info(f"日志路径: {log_path}")

    def feature_engineering(self):
        logger.info("开始特征工程...")

        df = pd.concat([self.history_data, self.future_data], axis=0)

        # 时间格式
        df["time"] = pd.to_datetime(df["time"])

        # 时间特征
        df["hour"] = df["time"].dt.hour
        df["month"] = df["time"].dt.month
        df["is_workday"] = (df["time"].dt.weekday < 5).astype(int)

        # 周期性编码
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # ===================== lag 特征 =====================
        lags = [1, 24, 48]

        for lag in lags:
            df[f"lag_{lag}"] = df[self.target_col].shift(lag)

        # ===================== rolling 特征 =====================
        df["rolling_mean_24"] = df[self.target_col].rolling(24).mean()
        df["rolling_std_24"] = df[self.target_col].rolling(24).std()

        # ===== 严格只保留预测区间 =====
        df = df[df["time"] >= self.split_time]

        # 删除空值
        df = df.dropna()

        # ===== 保存时间索引（关键）=====
        self.time_index = df["time"].copy()

        # 删除时间列
        df = df.drop(columns=["time"])

        self.X = df.drop(columns=[self.target_col])
        self.y = df[self.target_col]

        logger.success(f"特征工程完成 | 样本数: {self.X.shape}")

    def predict(self):
        logger.info("开始预测...")

        model_file = os.path.join(self.model_path, "xgb_model.pkl")
        model = joblib.load(model_file)

        self.feature_engineering()

        y_pred = model.predict(self.X)

        result = pd.DataFrame({
            "time": self.time_index.values,
            "y_true": self.y.values,
            "y_pred": y_pred
        })

        metrics = evaluate(self.y, y_pred)


        # ===== 打印到日志 =====
        logger.info(
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAPE={metrics['MAPE']:.2f}% | "
            f"R2={metrics['R2']:.4f}"
        )

        # ===== 保存 metrics（强烈建议）=====
        metrics_path = os.path.join(self.model_path, "metrics.json")
        import json
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # ===== 绘图 =====
        plot_prediction_with_time(result)

        # ===== 保存预测结果 =====
        save_path = os.path.join(self.model_path, "prediction.csv")
        result.to_csv(save_path, index=False)

        logger.success(f"预测完成，结果已保存至: {save_path}")


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 防止除0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    r2 = r2_score(y_true, y_pred)

    print("===== 评估指标 =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def plot_prediction_with_time(df):
    # 创建画布
    fig = plt.figure(figsize=(12, 5))

    # 绘制曲线：真实值、预测值
    plt.plot(df["time"], df["y_true"], label="真实值")
    plt.plot(df["time"], df["y_pred"], label="预测值", alpha=0.7)

    plt.title("预测值与真实值对比（时间序列）")
    plt.xlabel("时间")
    plt.ylabel("电力负荷")

    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    fig.savefig("./img/prediction.png", bbox_inches="tight", dpi=300)
    plt.show()



if __name__ == '__main__':
    p = PowerLoadPredictor(data_path='data', model_path='model', target_col='power_load')
    p.predict()

