import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import joblib

# ==================== 数据准备 ====================
# 加载数据（请替换实际文件路径）
train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")


# 特征与目标分离（假设第一列为ID，第二列为目标变量）
def prepare_data(df):
    X = df.iloc[:, 2:]  # 特征列
    y = df.iloc[:, 1]  # 目标列
    return X, y


X_train, y_train = prepare_data(train_set)
X_test, y_test = prepare_data(test_set)

# 标签编码
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# ==================== 参数配置 ====================
param_dist = {
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.3),
    'gamma': uniform(0, 1),
    'min_child_weight': randint(1, 6),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_lambda': uniform(0, 5),
    'reg_alpha': uniform(0, 2),
    'n_estimators': randint(100, 500)  }

# ==================== 模型配置 ====================
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    use_label_encoder=False,
    eval_metric='merror',
    early_stopping_rounds=50,
    random_state=42
)

# 配置交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================== 参数搜索 ====================
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=1000,
    cv=cv,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

# 划分验证集用于早停
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42
)

# 训练模型
print("开始参数优化...")
search.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ==================== 模型评估 ====================
best_model = search.best_estimator_


def evaluate_model(model, X, y, dataset_name):
    """综合模型评估"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    print(f"\n=== {dataset_name} 评估 ===")
    print(f"准确率: {accuracy_score(y, y_pred):.4f}")
    print("分类报告:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    # 计算AUC
    if len(le.classes_) > 2:
        auc = roc_auc_score(y, y_proba, multi_class='ovr')
    else:
        auc = roc_auc_score(y, y_proba[:, 1])
    print(f"AUC: {auc:.4f}")


# 评估训练集和测试集
evaluate_model(best_model, X_train, y_train_enc, "训练集")
evaluate_model(best_model, X_test, y_test_enc, "测试集")

