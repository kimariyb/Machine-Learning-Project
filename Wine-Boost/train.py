import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv("./data/train.csv")
    df.info()

    # 将数据分为 X 和 y
    X = df.drop('quality', axis=1)
    y = df["quality"]

    le = LabelEncoder()
    y = le.fit_transform(y)

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

    xgb = XGBClassifier(random_state=42, objective='multi:softmax',
                        eval_metric='mlogloss',num_class=len(np.unique(y)))

    xgb_param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 150, 200],
        'subsample': [0.8, 1.0]
    }

    ada = AdaBoostClassifier(algorithm='SAMME',random_state=42)
    ada_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2],
    }

    gb = GradientBoostingClassifier(random_state=42)
    gb_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    }


    # 网格搜索初始化
    grid = GridSearchCV(
        estimator=gb,
        param_grid=gb_param_grid,
        scoring='roc_auc_ovr_weighted',  # 多分类 ROC-AUC
        cv=5,
        n_jobs=8,  # 调用所有CPU核心
        verbose=1,
    )

    print("\n开始网格搜索训练...")
    grid.fit(X_train_scaled, y_train)

    # 获取最优模型
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_scaled)  # 预测类别
    y_pred_proba = best_model.predict_proba(X_test_scaled)  # 预测概率（多分类）


    # 多分类指标（必须加 weighted，适配不平衡分类）
    print(f"训练集最优分数: {grid.best_score_:.4f}")
    print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"测试集精确率: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"测试集召回率: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"测试集F1分数: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"测试集ROC-AUC: {roc_auc_score(y_test, y_pred_proba, multi_class='ovr'):.4f}")

    # 混淆矩阵 & 分类报告
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    # 最优参数
    print("\n最优超参数：")
    print(grid.best_params_)


# XGBoost
# 训练集最优分数: 0.8338
# 测试集准确率: 0.6750
# 测试集精确率: 0.6642
# 测试集召回率: 0.6750
# 测试集F1分数: 0.6655
# 测试集ROC-AUC: 0.8356

# AdaBoost
# 训练集最优分数: 0.7271
# 测试集准确率: 0.5625
# 测试集精确率: 0.4618
# 测试集召回率: 0.5625
# 测试集F1分数: 0.5069
# 测试集ROC-AUC: 0.6968

# GBDT
# 训练集最优分数: 0.8122
# 测试集准确率: 0.6250
# 测试集精确率: 0.6144
# 测试集召回率: 0.6250
# 测试集F1分数: 0.6181
# 测试集ROC-AUC: 0.8196
