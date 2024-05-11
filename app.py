
from flask import Flask, redirect, request, render_template, url_for
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from joblib import load

app = Flask(__name__)
app.config["SECRET_KEY"] = "quanhoangduong"

# Load the trained RandomForest model



model = load('svm_model.joblib')

def get_predict(list):
    # Đọc dữ liệu
    dataset = pd.read_csv('weatherAUS.csv')
    X = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
    
    # Chuyển đổi phần tử mới thành một mảng NumPy với kiểu dữ liệu là 'object'
    new_element_array = np.array(list, dtype=object)
    
    # Thêm phần tử mới vào mảng
    X = np.append(X, [new_element_array], axis=0)

    
    # Định dạng lại dữ liệu
    imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    X = imputer.fit_transform(X)
    # data = datasets.load_breast_cancer()
    # X = data.data
    # y = data.target
   
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

   



# Define route for home
@app.route('/', methods=["POST", "GET"])
def home():
    if request.method=="POST":
    # Get input data from the form
        min_temp = float(request.form['min_temp']) if request.form['min_temp'] != 'NA' else np.nan
        max_temp = float(request.form['max_temp']) if request.form['max_temp'] != 'NA' else np.nan
        rainfall = float(request.form['rainfall']) if request.form['rainfall'] != 'NA' else np.nan
        wind_gust_dir = request.form['wind_gust_dir']
        wind_gust_speed = float(request.form['wind_gust_speed']) if request.form['wind_gust_speed'] != 'NA' else np.nan
        wind_dir_9am = request.form['wind_dir_9am']
        wind_dir_3pm = request.form['wind_dir_3pm']
        wind_speed_9am = float(request.form['wind_speed_9am']) if request.form['wind_speed_9am'] != 'NA' else np.nan
        wind_speed_3pm = float(request.form['wind_speed_3pm']) if request.form['wind_speed_3pm'] != 'NA' else np.nan
        humidity_9am = float(request.form['humidity_9am']) if request.form['humidity_9am'] != 'NA' else np.nan
        humidity_3pm = float(request.form['humidity_3pm']) if request.form['humidity_3pm'] != 'NA' else np.nan
        pressure_9am = float(request.form['pressure_9am']) if request.form['pressure_9am'] != 'NA' else np.nan
        pressure_3pm = float(request.form['pressure_3pm']) if request.form['pressure_3pm'] != 'NA' else np.nan
        cloud_9am = float(request.form['cloud_9am']) if request.form['cloud_9am'] != 'NA' else np.nan
        cloud_3pm = float(request.form['cloud_3pm']) if request.form['cloud_3pm'] != 'NA' else np.nan
        temp_9am = float(request.form['temp_9am']) if request.form['temp_9am'] != 'NA' else np.nan
        temp_3pm = float(request.form['temp_3pm']) if request.form['temp_3pm'] != 'NA' else np.nan
        rain_today = request.form['rain_today']
        
        # Create a DataFrame from the input data
        input_data = []
        input_data.append('Albury')
        input_data.append(min_temp)
        input_data.append(max_temp)
        input_data.append(rainfall)
        # input_data.append(evaporation)
        # input_data.append(sunshine)
        input_data.append(wind_gust_dir)
        input_data.append(wind_gust_speed)
        input_data.append(wind_dir_9am)
        input_data.append(wind_dir_3pm)
        input_data.append(wind_speed_9am)
        input_data.append(wind_speed_3pm)
        input_data.append(humidity_9am)
        input_data.append(humidity_3pm)
        input_data.append(pressure_9am)
        input_data.append(pressure_3pm)
        input_data.append(cloud_9am)
        input_data.append(cloud_3pm)
        input_data.append(temp_9am)
        input_data.append(temp_3pm)
        input_data.append(rain_today)

        
        
    
        # # Perform prediction
        prediction = get_predict(input_data)
        if prediction[0]==0:
            return redirect(url_for("troikhongmua"))
        else:
            return redirect(url_for("troimua"))
    else:
        return render_template('index.html')
    
@app.route('/troimua', methods=[ "GET"])
def troimua():
    return render_template("troimua.html")

@app.route('/troikhongmua', methods=[ "GET"])
def troikhongmua():
    return render_template("troikhongmua.html")


if __name__ == "__main__":
    app.run(debug=True)
