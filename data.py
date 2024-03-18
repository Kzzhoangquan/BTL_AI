# from sklearn import datasets

# # Tải dữ liệu
# data = datasets.load_breast_cancer()

# data = datasets.load_breast_cancer()
# X = data.data
# y = data.target

# print(X)
# print(y)
import numpy as np

# Tạo một mảng hai chiều
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Chuyển thành mảng một chiều
arr_1d = arr_2d.astype('int64').flatten()


print(arr_1d)


