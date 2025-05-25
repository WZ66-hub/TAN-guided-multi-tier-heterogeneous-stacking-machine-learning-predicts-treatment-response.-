import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import time

# ==================== 数据准备 ====================

train_set = pd.read_csv("train.csv")
inter_validation_set = pd.read_csv("inter.csv")


# 特征与目标分离（假设第一列为ID，第二列为目标变量）
def prepare_data(df, features_start=2, target_idx=1):
    X = df.iloc[:, features_start:]
    y = df.iloc[:, target_idx]
    return X, y


X_train, y_train = prepare_data(train_set)
X_inter_val, y_inter_val = prepare_data(inter_validation_set)

# 合并训练集和验证集用于交叉验证
X_full = pd.concat([X_train, X_inter_val])
y_full = pd.concat([y_train, y_inter_val])


# 创建管道
pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),  # 添加标准化层
    ('logistic', LogisticRegression(random_state=42))
])

# 优化后的参数网格
param_grid = {
    'poly__degree': [1, 2],  # 限制多项式阶数防止过拟合
    'logistic__C': np.logspace(-3, 2, 6),
    'logistic__penalty': ['l1', 'l2'],
    'logistic__solver': ['liblinear', 'saga'],
    'logistic__max_iter': [1000],
    'logistic__class_weight': ['balanced', None]
}

# 计算类别权重（处理不平衡数据）
classes = np.unique(y_full)
weights = compute_class_weight('balanced', classes=classes, y=y_full)
class_weights = dict(zip(classes, weights))

# ==================== 网格搜索 ====================
# 配置5折分层交叉验证
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=3
)

# 训练模型（使用合并数据）
print("开始网格搜索...")
start_time = time.time()
grid_search.fit(X_full, y_full)
print(f"总耗时: {time.time() - start_time:.2f}秒")

# ==================== 模型评估 ====================
best_model = grid_search.best_estimator_


def comprehensive_evaluation(model, X, y, dataset_name):
    """综合模型评估函数"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    print(f"\n=== {dataset_name} 评估 ===")
    print(f"准确率: {accuracy_score(y, y_pred):.4f}")
    print(f"AUC值: {roc_auc_score(y, y_proba, multi_class='ovr'):.4f}")
    print("分类报告:")
    print(classification_report(y, y_pred))
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))


# 评估各数据集表现
comprehensive_evaluation(best_model, X_train, y_train, "训练集")
comprehensive_evaluation(best_model, X_inter_val, y_inter_val, "中间验证集")

