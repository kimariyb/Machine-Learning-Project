import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv("./data/breast-cancer-wisconsin.csv")
    df.info()

    # 清洗数据，将不正确的数据替换为 NaN
    df.replace("?", np.nan, inplace=True)

    # 将包含 NaN 的样本去除
    df.dropna(inplace=True)
    df = df.astype(float)

    df.info()

    # 将数据分为 X 和 y
    X = df.drop(["Class", "Sample code number"], axis=1)
    y = df["Class"]

    # 打印验证（可选）
    print("\n===== 特征X形状 =====", X.shape)
    print("===== 标签y形状 =====", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify=y 保证分层抽样，标签分布一致
    )
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建模型
    clf = LogisticRegressionCV(cv=5, verbose=1, n_jobs=8, scoring='roc_auc')
    clf.fit(X_train_scaled, y_train)

    # 对测试集预测
    y_pred = clf.predict(X_test_scaled)

    # 8. 模型评估（输出所有核心指标）
    print("\n" + "=" * 60)
    print("                模型评估结果")
    print("=" * 60)

    # 基础指标
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1分数 (F1 Score): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"ROC-AUC 分数: {roc_auc_score(y_test, y_pred):.4f}")

    # 混淆矩阵
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 分类报告（包含每个类别的指标）
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 输出模型最优参数
    print(f"\n模型最优正则化参数 C: {clf.C_[0]:.4f}")


# 准确率 (Accuracy): 0.9635
# 精确率 (Precision): 0.9650
# 召回率 (Recall): 0.9635
# F1分数 (F1 Score): 0.9637
# ROC-AUC 分数: 0.9671