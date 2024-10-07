import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# โหลดข้อมูล Covertype dataset
data = fetch_covtype()
X = data.data
y = data.target

# แบ่งข้อมูลเป็น training set และ test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เซฟข้อมูลแถวแรกของ test set ลงไฟล์ CSV (รวม target)
first_row_data = np.column_stack((X_test[0].reshape(1, -1), y_test[0]))  # รวมฟีเจอร์และ target ของแถวแรก
first_row_df = pd.DataFrame(first_row_data, columns=[f'feature_{i}' for i in range(X_test.shape[1])] + ['target'])  # ตั้งชื่อคอลัมน์
first_row_df.to_csv('test_first_row_data.csv', index=False)

print("ข้อมูลแถวแรกของ test set ถูกบันทึกแล้วในไฟล์ test_first_row_data.csv")

# ลบข้อมูลแถวแรกออกจาก X_test และ y_test
X_test = np.delete(X_test, 0, axis=0)  # ลบแถวแรกออกจาก X_test
y_test = np.delete(y_test, 0, axis=0)  # ลบแถวแรกออกจาก y_test

# บันทึกข้อมูล test set ที่เหลือ ลงไฟล์ CSV (ไม่รวม target)
test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])  # เก็บเฉพาะฟีเจอร์
test_df.to_csv('test_data.csv', index=False)

print("ข้อมูล test set ที่เหลือ (เฉพาะฟีเจอร์) ถูกบันทึกแล้วในไฟล์ test_data.csv")

# สร้าง Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# ฝึกโมเดล
clf.fit(X_train, y_train)

# ไม่บันทึกโมเดลลงไฟล์ .sav อีกต่อไป
print("โมเดลถูกฝึกแล้ว แต่ไม่ได้บันทึกไฟล์ random_forest_model.sav")

# บันทึกโมเดลเป็นไฟล์ .sav
with open('random_forest_model.sav', 'wb') as model_file:
    pickle.dump(clf, model_file)

print("โมเดลถูกบันทึกแล้วในไฟล์ random_forest_model.sav")
