from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from joblib import load


# Load the trained RandomForest model



model = load('svm_model.joblib')

def get_predict(list):
    # Đọc dữ liệu
    dataset = pd.read_csv('weatherAUS.csv')
    X = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
    print(X[0])
    
    # Chuyển đổi phần tử mới thành một mảng NumPy với kiểu dữ liệu là 'object'
    new_element_array = np.array(list, dtype=object)
    
    # Thêm phần tử mới vào mảng
    X = np.append(X, [new_element_array], axis=0)

    
    # Định dạng lại dữ liệu
    imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    X = imputer.fit_transform(X)
   
   
    le1 = LabelEncoder()
    X[:,0] = le1.fit_transform(X[:,0])
    le2 = LabelEncoder()
    X[:,4] = le2.fit_transform(X[:,4])
    le3 = LabelEncoder()
    X[:,6] = le3.fit_transform(X[:,6])
    le4 = LabelEncoder()
    X[:,7] = le4.fit_transform(X[:,7])
    le5 = LabelEncoder()
    X[:,-1] = le5.fit_transform(X[:,-1])


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
  
    # Lấy mẫu là phần tử cuối cùng trong mảng
    sample = [X[-1].tolist()]
    result = model.predict(sample)
    # result = model.predict_proba(sample)
    return result


# Ví dụ

sample = ["1",2,3,4,"7","8","9","10",11,12,13,14,15,16,17,18,19,20,"21"]
print(get_predict(sample))
# Albury,13.4,22.9,0.6,NA,NA,W,44,W,WNW,20,24,71,22,1007.7,1007.1,8,NA,16.9,21.8,No,