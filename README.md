# Machine Learning Project

## 1. Iris-KNN 鸢尾花分类

- 任务：经典**多分类**，识别鸢尾花品种
- 模型：`KNeighborsClassifier`
- 训练结果：
  - Accuracy：`0.9556 `
  - Precision：`0.9608 `
  - Recall：`0.9556 `
  - F1-score：`0.9554`

## 2. Cancer-Logic 癌症分类

- 任务：医疗**二分类**，判断肿瘤良恶性
- 模型：`LogisticRegression`
- 训练结果：
  - Accuracy：`0.9635`
  - Precision：`0.9650`
  - Recall：`0.9635`
  - F1-score：`0.9637`

    
## 3. Titanic-DTree/RF 泰坦尼克号分类

- 任务：乘客生存预测二分类
- 模型：`DecisionTreeClassifier` / `RandomForestClassifier`
- 决策树训练结果：
  - Accuracy：`0.7537`
  - Precision：`0.6907`
  - Recall：`0.6505`
  - F1-score：`0.6700`
- 随机森林训练结果：
  - Accuracy：`0.7649`
  - Precision：`0.7041`
  - Recall：`0.6699`
  - F1-score：`0.6866`

## 4. Wine-Boost 红酒分类

- 任务：红酒质量等级**多分类**
- 模型：`AdaBoostClassifier` / `GradientBoostingClassifier` / `XGBClassifier` 
- AdaBoost 训练结果：
  - Accuracy：`0.5625`
  - Precision：`0.4618`
  - Recall：`0.5625`
  - F1-score：`0.5069`
- GBDT 训练结果：
  - Accuracy：`0.6250`
  - Precision：`0.6144`
  - Recall：`0.6250`
  - F1-score：`0.6181`
- XGBoost 训练结果：
  - Accuracy：`0.6750`
  - Precision：`0.6642`
  - Recall：`0.6750`
  - F1-score：`0.6655`


## 5. PowerForecasting-XGBoost 电力预测回归
- 任务：电力负荷时序**回归预测**
- 模型：`XGBRegressor`
- 训练结果：
  - MAE：`54.379`
  - RMSE：`89.187`
  - MAPE：`5.445`
  - R2：`0.801`

