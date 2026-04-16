import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from loguru import logger
from utils import data_processing
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 兼容Windows/Linux/Mac
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (12, 16)  # 设置画布整体大小


class PowerForecastingModel:
    def __init__(self, target_col="power_load"):
        self.target_col = target_col
        self.model = None

        self.results_dir = "./model"
        os.makedirs(self.results_dir, exist_ok=True)

        self._setup_logger()
        self._load_data()

        logger.success("PowerForecastingModel 初始化完成")


    def _load_data(self):
        data_dir = "./data"
        self.train_path = os.path.join(data_dir, "train.csv")
        self.test_path = os.path.join(data_dir, "test.csv")

        self.train_data = data_processing(self.train_path)
        self.test_data = data_processing(self.test_path)

        logger.info(f"训练集: {self.train_data.shape}")
        logger.info(f"测试集: {self.test_data.shape}")

    def feature_engineering(self):
        logger.info("开始特征工程...")

        df_train = self.train_data.copy()
        df_test = self.test_data.copy()

        # 时间格式
        df_train["time"] = pd.to_datetime(df_train["time"])
        df_test["time"] = pd.to_datetime(df_test["time"])

        # 时间特征
        for df in [df_train, df_test]:
            df["hour"] = df["time"].dt.hour
            df["month"] = df["time"].dt.month
            df["is_workday"] = (df["time"].dt.weekday < 5).astype(int)

            # 周期性编码
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # ===================== lag 特征 =====================
        lags = [1, 24, 48]

        for lag in lags:
            df_train[f"lag_{lag}"] = df_train[self.target_col].shift(lag)
            df_test[f"lag_{lag}"] = df_test[self.target_col].shift(lag)

        # ===================== rolling 特征 =====================
        df_train["rolling_mean_24"] = df_train[self.target_col].rolling(24).mean()
        df_train["rolling_std_24"] = df_train[self.target_col].rolling(24).std()

        df_test["rolling_mean_24"] = df_test[self.target_col].rolling(24).mean()
        df_test["rolling_std_24"] = df_test[self.target_col].rolling(24).std()

        # 删除空值
        df_train = df_train.dropna()
        df_test = df_test.dropna()

        # 删除时间列
        df_train = df_train.drop(columns=["time"])
        df_test = df_test.drop(columns=["time"])

        self.train_data = df_train
        self.test_data = df_test

        self._split_features_labels()

        logger.success(f"特征工程完成 | 特征数: {self.X.shape[1]}")

    def _setup_logger(self):
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

        log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    def _split_features_labels(self):
        self.X = self.train_data.drop(columns=[self.target_col])
        self.y = self.train_data[self.target_col]

        if self.target_col in self.test_data.columns:
            self.X_test = self.test_data.drop(columns=[self.target_col])
            self.y_test = self.test_data[self.target_col]
        else:
            self.X_test = self.test_data
            self.y_test = None

    def train(self, cv=5):
        logger.info("开始训练（TimeSeries CV）...")

        self.model = XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
        )

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 150, 200],
            'subsample': [0.8, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits=cv)

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X, self.y)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_rmse = -grid_search.best_score_

        logger.success(f"最优参数: {self.best_params}")
        logger.success(f"CV RMSE: {self.best_rmse:.4f}")

        # 保存模型
        model_path = os.path.join(self.results_dir, "xgb_model.pkl")
        joblib.dump(self.model, model_path)
        logger.success(f"模型已保存: {model_path}")

    def evaluate(self):
        logger.info("模型评估...")

        if self.y_test is not None:
            preds = self.model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            logger.info(f"测试集 RMSE: {rmse:.4f}")
        else:
            logger.info("无测试集标签，跳过评估")

    def feature_importance(self):
        importance = pd.DataFrame({
            "feature": self.X.columns,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        logger.info(f"Top10 特征:\n{importance.head(10)}")

    def run(self):
        self.feature_engineering()
        self.train()
        self.evaluate()
        self.feature_importance()




def analyze_data(data: pd.DataFrame, time_col: str = "time", target_col: str = "power_load"):
    """
    功率/负荷数据探索性分析
    绘制4张核心分析图：负荷分布、月份趋势、小时趋势、工作日/休息日对比
    :param data: 输入数据集 (DataFrame)
    :param time_col: 时间列列名（必须是datetime类型或可解析为时间）
    :param target_col: 负荷/功率目标列列名
    """
    try:
        # ===================== 1. 数据基础校验 & 信息查看 =====================
        logger.info("=" * 50)
        logger.info("开始执行数据探索性分析")
        logger.info(f"数据形状: {data.shape} | 时间列: {time_col} | 目标列: {target_col}")

        # 校验关键列是否存在
        if time_col not in data.columns or target_col not in data.columns:
            raise ValueError(f"数据中缺少必要列：请检查时间列[{time_col}]或目标列[{target_col}]")

        # 转换时间列（自动解析日期）
        data[time_col] = pd.to_datetime(data[time_col])

        logger.info("数据统计描述:")
        logger.info(f"\n{data[target_col].describe()}")

        # ===================== 2. 提取时间特征（核心） =====================
        data["month"] = data[time_col].dt.month  # 月份 (1-12)
        data["hour"] = data[time_col].dt.hour  # 小时 (0-23)
        data["weekday"] = data[time_col].dt.weekday  # 星期 (0=周一, 6=周日)
        data["workday"] = data["weekday"].apply(lambda x: "工作日" if x < 5 else "休息日")

        # 分组计算统计值
        month_load = data.groupby("month")[target_col].mean()  # 月均负荷
        hour_load = data.groupby("hour")[target_col].mean()  # 小时均负荷
        workday_load = data.groupby("workday")[target_col].mean()  # 工作日/休息日负荷

        # ===================== 3. 绘制4张子图 =====================
        fig = plt.figure()
        fig.suptitle("功率/负荷数据探索性分析", fontsize=16, fontweight="bold")  # 总标题

        # 子图1：负荷整体分布直方图
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.hist(data[target_col], bins=50, rwidth=0.9, color="#1f77b4", alpha=0.7)
        ax1.set_title("负荷整体分布直方图", fontsize=12)
        ax1.set_xlabel("负荷值")
        ax1.set_ylabel("频次")
        ax1.grid(alpha=0.3)

        # 子图2：月份-平均负荷趋势图
        ax2 = fig.add_subplot(4, 1, 2)
        ax2.plot(month_load.index, month_load.values, marker="o", color="#ff7f0e", linewidth=2)
        ax2.set_title("月份与平均负荷趋势", fontsize=12)
        ax2.set_xlabel("月份")
        ax2.set_ylabel("平均负荷")
        ax2.set_xticks(np.arange(1, 13))  # 1-12月
        ax2.grid(alpha=0.3)

        # 子图3：小时-平均负荷趋势图
        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(hour_load.index, hour_load.values, marker="s", color="#2ca02c", linewidth=2)
        ax3.set_title("24小时平均负荷趋势", fontsize=12)
        ax3.set_xlabel("小时")
        ax3.set_ylabel("平均负荷")
        ax3.set_xticks(np.arange(0, 24, 2))  # 0-23点，间隔2
        ax3.grid(alpha=0.3)

        # 子图4：工作日/休息日平均负荷对比图
        ax4 = fig.add_subplot(4, 1, 4)
        ax4.bar(workday_load.index, workday_load.values, color=["#d62728", "#9467bd"], alpha=0.8)
        ax4.set_title("工作日 vs 休息日平均负荷对比", fontsize=12)
        ax4.set_ylabel("平均负荷")
        ax4.grid(alpha=0.3, axis="y")

        # 自动调整子图间距，防止重叠
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("./img/analyze_data.png", dpi=300)
        # 显示图片
        plt.show()

        logger.success("数据探索性分析完成！")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"数据分析失败: {str(e)}")
        raise





if __name__ == '__main__':
    pf = PowerForecastingModel()
    pf.run()
