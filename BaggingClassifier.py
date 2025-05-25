import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import clone
import joblib



train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")


X_train = train_set.iloc[:, 2:]
y_train = train_set.iloc[:, 1]
X_test = test_set.iloc[:, 2:]
y_test = test_set.iloc[:, 1]


param_grid = {
    'estimator__max_depth': [3, 5, 7, 9, None],
    'estimator__min_samples_split': [2, 5, 10, 15],
    'estimator__min_samples_leaf': [1, 2, 4],
    'n_estimators': [50, 100, 200],
    'max_samples': [0.6, 0.8, 1.0],
    'max_features': [0.6, 0.8, 1.0]
}


base_tree = DecisionTreeClassifier(random_state=42)
model = BaggingClassifier(
    estimator=base_tree,
    random_state=42,
    n_jobs=-1  # 启用并行计算
)

# ==================== 网格搜索 ====================
# 使用5折交叉验证
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)

# 执行网格搜索（只在训练集上）
grid_search.fit(X_train, y_train)

# ==================== 最佳模型评估 ====================
best_model = grid_search.best_estimator_


# 定义综合评估函数
def comprehensive_evaluation(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='macro'),
        'Recall': recall_score(y, y_pred, average='macro'),
        'F1': f1_score(y, y_pred, average='macro'),
        'AUC': roc_auc_score(y, y_proba, multi_class='ovr'),
        'Confusion_Matrix': confusion_matrix(y, y_pred)
    }
    return metrics


# 评估各数据集表现
train_metrics = comprehensive_evaluation(best_model, X_train, y_train)
test_metrics = comprehensive_evaluation(best_model, X_test, y_test)

# ==================== 结果输出 ====================
print("\n=== 最佳参数 ===")
print(grid_search.best_params_)

print("\n=== 训练集评估 ===")
print(f"准确率: {train_metrics['Accuracy']:.4f}")
print(f"精确率: {train_metrics['Precision']:.4f}")
print(f"召回率: {train_metrics['Recall']:.4f}")
print(f"F1分数: {train_metrics['F1']:.4f}")
print(f"AUC值: {train_metrics['AUC']:.4f}")

print("\n=== 测试集评估 ===")
print(f"准确率: {test_metrics['Accuracy']:.4f}")
print(f"精确率: {test_metrics['Precision']:.4f}")
print(f"召回率: {test_metrics['Recall']:.4f}")
print(f"F1分数: {test_metrics['F1']:.4f}")
print(f"AUC值: {test_metrics['AUC']:.4f}")
print("混淆矩阵:")
print(test_metrics['Confusion_Matrix'])

