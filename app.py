from flask import Flask, request, jsonify
import pickle
import numpy as np

# Khởi tạo Flask
app = Flask(__name__)

# Tải mô hình đã tối ưu
with open('./optimized_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ yêu cầu
    data = request.json
    features = np.array(data['features']).reshape(1, -1)  # Chuyển đổi dữ liệu thành định dạng phù hợp
    
    # Dự đoán
    prediction = model.predict(features)
    
    # Trả về dự đoán
    return jsonify({'predicted_value': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
