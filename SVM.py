import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform

# 设置可重复性种子
SEED = 42
np.random.seed(SEED)


# 数据准备（需要补全实际列索引）
def load_data(train_path, test_path):
    """加载并预处理数据"""
    # 读取数据
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)


    X_train = train_set.iloc[:, 2:]
    y_train = train_set.iloc[:, 1]
    X_test = test_set.iloc[:, 2:]
    y_test = test_set.iloc[:, 1]

    # 标签编码
    le = LabelEncoder()
    le.fit([1, 2, 4])  # 根据实际类别标签调整
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, le.classes_


# 加载数据（请替换实际文件路径）
X_train, y_train, X_test, y_test, class_names = load_data("train.csv", "test.csv")

# 定义参数空间
param_dist = {
    'svc__C': loguniform(1e-3, 1e3),  # 正则化参数
    'svc__gamma': loguniform(1e-4, 1e1),  # 核函数系数
    'svc__kernel': ['rbf', 'poly', 'sigmoid'],
    'svc__degree': randint(1, 5),  # 多项式阶数
    'svc__class_weight': [None, 'balanced']  # 类别权重
}

# 创建流水线（包含标准化和SVM）
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 标准化已在前处理完成，此处可移除
    ('svc', SVC(probability=True, random_state=SEED))
])

# 配置交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 配置随机搜索
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=200,  # 从5000调整为更合理的数值
    cv=cv,
    scoring='accuracy',
    verbose=2,
    random_state=SEED,
    n_jobs=-1,
    return_train_score=True
)

# 执行参数搜索（仅在训练集上）
print("开始参数优化...")
random_search.fit(X_train, y_train)

# 获取最佳模型
best_model = random_search.best_estimator_


# 最终模型评估
def evaluate_model(model, X, y, dataset_name):
    """综合模型评估"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    print(f"\n=== {dataset_name} 评估结果 ===")
    print(f"准确率: {accuracy_score(y, y_pred):.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))
    print("分类报告:")
    print(classification_report(y, y_pred, target_names=class_names))



# 评估训练集和测试集
evaluate_model(best_model, X_train, y_train, "训练集")
evaluate_model(best_model, X_test, y_test, "测试集")

