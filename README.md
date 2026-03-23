# Loan Default Risk Prediction

ระบบทำนายความเสี่ยงการผิดนัดชำระสินเชื่อ โดยใช้ Machine Learning
วิเคราะห์ข้อมูลทางการเงินของผู้ขอสินเชื่อ

## Web Application
🔗 [เปิด App ได้ที่นี่](https://repoproject-rbdxn7cgyjlj3dsgzwf32n.streamlit.app/)

## Dataset
- **ที่มา:** Give Me Some Credit (Kaggle)
- **ขนาด:** 150,000 rows, 10 features
- **เป้าหมาย:** ทำนายว่าลูกค้าจะผิดนัดชำระหนี้ใน 2 ปีข้างหน้าหรือไม่

## โมเดลที่ใช้
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.7895 |
| Random Forest | 0.8349 |
| Gradient Boosting | 0.8687 |

โมเดลที่ดีที่สุดคือ **Gradient Boosting** ROC-AUC = 0.8687

## โครงสร้างไฟล์
```
loan-default-prediction/
├── app.py              ← Streamlit application
├── train_model.py      ← Training script
├── requirements.txt    ← Dependencies
├── models/
│   └── best_model.pkl  ← Trained model
└── notebooks/
    └── DS_PRO.ipynb    ← EDA และ Model Development
```

## วิธีรันบนเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```
