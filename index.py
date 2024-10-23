from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import pickle
import numpy as np

# Đọc dữ liệu từ file CSV (đảm bảo thay đúng tên file)
data = pd.read_csv('final_data.csv')

# Xem một vài dòng đầu tiên
print(data.head())

# Chọn các đặc trưng đầu vào (features) và nhãn (target)
features = data[['age', 'appearance', 'goals', 'assists', 
                 'yellow cards', 'goals conceded', 'clean sheets', 
                 'minutes played', 'highest_value', 'games_injured']]
target = data['current_value']




# Chia dữ liệu thành tập train và test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình hồi quy tuyến tính ( chỉ sử dụng XGBoost)
# model = LinearRegression()
# model.fit(X_train, y_train)

# Khởi tạo tham số cho XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Tìm kiếm tham số tối ưu với RandomizedSearchCV
xgb = XGBRegressor(random_state=42, tree_method='hist')
xgb_search = RandomizedSearchCV(xgb, param_distributions=xgb_params, n_iter=50, cv=5, n_jobs=-1, random_state=42)
xgb_search.fit(X_train, y_train)

# # Lưu mô hình đã tối ưu với pickle
# best_xgb_model = xgb_search.best_estimator_
# with open('optimized_xgb_model.pkl', 'wb') as f:
#     pickle.dump(best_xgb_model, f)

# Đánh giá mô hình
y_pred = best_xgb_model.predict(X_test)
print(f'XGBoost - MSE: {mean_squared_error(y_test, y_pred)}, R²: {r2_score(y_test, y_pred)}')

# Dự đoán giá trị của một cầu thủ mới (ví dụ)
new_player = [[30, 15, 0, 0, 0, 1.24, 0.33, 9390, 70000000, 42]]  # Chỉnh theo dữ liệu mẫu
predicted_value = best_xgb_model.predict(new_player)

print(f'Giá trị dự đoán của cầu thủ mới: {predicted_value[0]} triệu euro')
