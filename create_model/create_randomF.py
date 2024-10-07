import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle  # นำเข้าไลบรารี pickle สำหรับบันทึกโมเดล

# อ่านข้อมูลจากไฟล์
data = pd.read_csv('cleaned_fat_data.csv')

X = data.drop('Nobeyesdad__CODE', axis=1)  # ลบคอลัมน์ที่เป็นเป้าหมายออกจาก X
y = data['Nobeyesdad__CODE']  # ใช้คอลัมน์ Nobeyesdad__CODE เป็น y

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล RandomForest
rf_model = RandomForestClassifier(n_estimators=100)  
rf_model.fit(X_train, y_train)  # เทรนโมเดล
# บันทึกโมเดลลงในไฟล์ด้วย pickle
with open('random_forest.sav', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

print("Model saved as 'random_forest.sav' using pickle")
