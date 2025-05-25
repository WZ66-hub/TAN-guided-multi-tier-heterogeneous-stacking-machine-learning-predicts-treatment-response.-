import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== 数据准备 ====================
# 加载数据（请替换实际文件路径）
train_set = pd.read_csv("train.csv")
inter_validation_set = pd.read_csv("inter.csv")


# 特征与目标分离（假设第一列为ID，第二列为目标变量）
def prepare_data(df, features_start=2, target_idx=1):
    X = df.iloc[:, features_start:]
    y = df.iloc[:, target_idx]
    return X, y


X_train, y_train = prepare_data(train_set)
X_inter_val, y_inter_val = prepare_data(inter_validation_set)

# 合并训练集和中间验证集用于交叉验证
X_full = pd.concat([X_train, X_inter_val])
y_full = pd.concat([y_train, y_inter_val])

# 类别编码
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_full_enc = le.transform(y_full)

# ==================== 参数配置 ====================
# 基础参数
base_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

# 优化后的参数网格
param_grid = {
    'num_leaves': [40, 60, 80],  # 简化范围
    'learning_rate': [0.01, 0.05, 0.1],  # 优化学习率范围
    'n_estimators': [200, 300],  # 配合早停机制
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}

# ==================== 模型配置 ====================
# 使用早停机制的回调函数
early_stop = lgb.early_stopping(stopping_rounds=50, verbose=False)
eval_metric = lgb.log_evaluation(period=100)

# 初始化模型
model = lgb.LGBMClassifier(**base_params)

# 配置分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================== 网格搜索 ====================
grid_search = GridSearchCV(
estimator = model,
param_grid = param_grid,
cv = cv,
scoring = 'accuracy',
verbose = 3,
n_jobs = -1
)

# 执行网格搜索（使用合并后的数据）
grid_search.fit(
X_full, y_full_enc,
callbacks = [early_stop, eval_metric]
)

# ==================== 最佳模型评估 ====================
best_model = grid_search.best_estimator_


def evaluate_model(model, X, y, dataset_name):
    """综合模型评估函数"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    print(f"\n=== {dataset_name} 评估结果 ===")
    print(f"准确率: {accuracy_score(y, y_pred):.4f}")
    print(f"加权F1: {f1_score(y, y_pred, average='weighted'):.4f}")
    print("分类报告:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))


# 评估各数据集
evaluate_model(best_model, X_train, y_train_enc, "训练集")
evaluate_model(best_model, X_inter_val, le.transform(y_inter_val), "中间验证集")


