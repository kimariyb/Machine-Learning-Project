import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv("./data/train.csv")
    df.info()

    # 将数据分为 X 和 y
    X = df[['Pclass', 'Sex', 'Age']]
    y = df["Survived"]

    # 处理异常值
    X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())

    # 对 x 进行 one-hot 编码处理
    X = pd.get_dummies(X)
    X.drop("Sex_female", axis=1, inplace=True)

    # 打印验证（可选）
    print("\n===== 特征X形状 =====", X.shape)
    print("===== 标签y形状 =====", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # stratify=y 保证分层抽样，标签分布一致
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建模型
    clf = RandomForestClassifier(verbose=1, random_state=42)

    param_grid = {
        # 集成数量
        'n_estimators': [20, 30, 50, 100, 150, 200],
        # 预剪枝
        'max_depth': [3, 4, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3],
        'criterion': ['gini', 'entropy'],
        # 单棵树后剪枝
        'ccp_alpha': [0, 0.001, 0.005, 0.01]
    }

    # 网格搜索
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=8,
        verbose=1,
    )

    grid.fit(X_train_scaled, y_train)

    # 取网格搜索最优模型
    best_model = grid.best_estimator_

    # 模型预测
    y_pred = best_model.predict(X_test_scaled)  # 预测类别（用于准确率、精确率等）
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # 预测正类概率（专用于ROC-AUC）

    # 模型评估结果输出
    print("\n" + "=" * 60)
    print("                决策树模型评估结果")
    print("=" * 60)

    # 基础分类指标
    print(f"训练集最优 ROC-AUC: {grid.best_score_:.4f}")
    print(f"测试集 准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"测试集 精确率 (Precision): {precision_score(y_test, y_pred):.4f}")
    print(f"测试集 召回率 (Recall): {recall_score(y_test, y_pred):.4f}")
    print(f"测试集 F1分数 (F1 Score): {f1_score(y_test, y_pred):.4f}")
    print(f"测试集 ROC-AUC 分数: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # 混淆矩阵
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 输出最优参数（含剪枝系数）
    print("\n" + "=" * 60)
    print("                最优超参数（含剪枝）")
    print("=" * 60)
    print(grid.best_params_)


# 训练集最优 ROC-AUC: 0.8683
# 测试集 准确率 (Accuracy): 0.7649
# 测试集 精确率 (Precision): 0.7041
# 测试集 召回率 (Recall): 0.6699
# 测试集 F1分数 (F1 Score): 0.6866
# 测试集 ROC-AUC 分数: 0.8480