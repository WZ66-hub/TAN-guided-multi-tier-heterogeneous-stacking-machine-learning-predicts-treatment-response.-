import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 读取数据（需补全文件路径）
train_set = pd.read_csv("train.csv")
test_set = pd.read_excel("test.xlsx")

# 修正列索引（根据实际数据结构调整）
X_train = train_set.iloc[:, 2:]
Y_train = train_set.iloc[:, 1]
X_test = test_set.iloc[:, 1:12]
Y_test = test_set.iloc[:, 12]

# 定义参数分布
param_dist = {
    'max_depth': randint(1, 50),
    'min_samples_split': randint(2, 30),
    'min_samples_leaf': randint(1, 20),
    'max_features': uniform(0.1, 0.9),
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'class_weight': ['balanced', None],
    'ccp_alpha': uniform(0, 0.05)
}

# 创建决策树分类器
dt = DecisionTreeClassifier(random_state=42)


random_search = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_dist,
    n_iter=1000,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)


random_search.fit(X_train, Y_train)

# 直接使用最佳模型
best_model = random_search.best_estimator_

# 输出最佳参数和交叉验证结果
print("\n=== 最佳参数 ===")
print("Best parameters:", random_search.best_params_)
print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

# 使用最佳模型进行测试集评估
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)

# 计算训练集准确率（可选）
train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)

# 输出最终结果
print("\n=== 最终评估结果 ===")
print(f"Final train accuracy: {train_accuracy:.4f}")
print(f"Final test accuracy: {test_accuracy:.4f}")

# 可选：输出模型特征重要性
if hasattr(best_model, 'feature_importances_'):
    print("\n=== 特征重要性 ===")
    for idx, importance in enumerate(best_model.feature_importances_):
        print(f"Feature {X_train.columns[idx]}: {importance:.4f}")