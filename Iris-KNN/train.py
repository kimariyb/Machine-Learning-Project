import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'


if __name__ == '__main__':
    # 获取数据集
    iris = load_iris()

    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 拼接成 df
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    print("数据集基本信息：")
    print(df.head())
    print("="*60)

    # 可视化数据结构
    g = sns.pairplot(df, hue='target', kind='scatter')
    g.fig.suptitle('鸢尾花特征两两关系分布图', y=1.02)
    g.savefig("./img/iris_pairplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 初始化 KNN 模型
    model = KNeighborsClassifier()

    # 网格搜索
    param_grid = {'n_neighbors': range(2, 21)}

    # 5折交叉验证，评分指标为准确率
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=8  # 调用所有CPU核心，加速搜索
    )
    grid_search.fit(X_train_scaled, y_train)

    print("网格搜索最优参数：", grid_search.best_params_)
    print("训练集交叉验证最优准确率：{:.4f}".format(grid_search.best_score_))
    print("="*60)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # 多分类用宏平均
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # 打印评估结果
    print("测试集模型评估指标：")
    print(f"准确率(Accuracy)：{accuracy:.4f}")
    print(f"精确率(Precision)：{precision:.4f}")
    print(f"召回率(Recall)：{recall:.4f}")
    print(f"F1分数：{f1:.4f}")
    print("=" * 60)

    # 打印详细分类报告
    print("详细分类报告：")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,        # 显示数字
        fmt="d",           # 整数格式
        cmap="Blues",      # 配色
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title('KNN 模型-鸢尾花混淆矩阵', fontsize=12)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    # 保存混淆矩阵图片
    plt.savefig("./img/iris_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


"""
============================================================
网格搜索最优参数： {'n_neighbors': 14}
训练集交叉验证最优准确率：0.9714
============================================================
测试集模型评估指标：
准确率(Accuracy)：0.9556
精确率(Precision)：0.9608
召回率(Recall)：0.9556
F1分数：0.9554
============================================================
详细分类报告：
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       0.88      1.00      0.94        15
   virginica       1.00      0.87      0.93        15

    accuracy                           0.96        45
   macro avg       0.96      0.96      0.96        45
weighted avg       0.96      0.96      0.96        45
"""