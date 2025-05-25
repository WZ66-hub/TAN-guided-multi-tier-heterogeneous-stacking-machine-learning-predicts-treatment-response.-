import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


train_set = pd.read_csv(".csv")
test_set = pd.read_csv(".csv")

x_train = train_set.iloc[:, ]
y_train = train_set.iloc[:, ]
x_test = test_set.iloc[:, ]
y_test = test_set.iloc[:, ]

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_

# 使用最佳参数重新构建随机森林模型
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(x_train, y_train)

# 在测试集上进行预测
y_pred = best_rf.predict(x_test)
y_pred_train = best_rf.predict(x_train)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Train_Accuracy: {train_accuracy}')


