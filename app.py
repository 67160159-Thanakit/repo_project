import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load('models/best_model.pkl')

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="🏦")
st.title("🏦 Loan Default Risk Predictor")
st.warning("⚠️ Disclaimer: ผลลัพธ์นี้เป็นเพียงการประมาณการ ไม่ใช่คำตัดสินใจทางการเงิน")
st.divider()

st.subheader("กรอกข้อมูลผู้ขอสินเชื่อ")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("รายได้ต่อเดือน (บาท)", min_value=0, value=30000)
    debt_ratio = st.slider("สัดส่วนหนี้ต่อรายได้", 0.0, 1.0, 0.3)
    revolving = st.slider("การใช้วงเงินบัตรเครดิต", 0.0, 1.0, 0.3)
    open_credit = st.number_input("จำนวนบัญชีสินเชื่อ", min_value=0, max_value=50, value=5)

with col2:
    late_30_59 = st.number_input("ค้างชำระ 30-59 วัน (ครั้ง)", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input("ค้างชำระ 60-89 วัน (ครั้ง)", min_value=0, max_value=20, value=0)
    late_90 = st.number_input("ค้างชำระ 90+ วัน (ครั้ง)", min_value=0, max_value=20, value=0)
    real_estate = st.number_input("สินเชื่ออสังหาริมทรัพย์", min_value=0, max_value=20, value=0)
    dependents = st.number_input("จำนวนผู้อยู่ในความดูแล", min_value=0, max_value=20, value=0)

st.divider()

if st.button("ประเมินความเสี่ยง", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': revolving,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': dependents
    }])

    prob = model.predict_proba(input_data)[0][1]
    risk_pct = prob * 100

    st.subheader("ผลการประเมิน")
    if prob < 0.3:
        st.success(f"✅ ความเสี่ยงต่ำ ({risk_pct:.1f}%)")
    elif prob < 0.6:
        st.warning(f"⚠️ ความเสี่ยงปานกลาง ({risk_pct:.1f}%)")
    else:
        st.error(f"🚨 ความเสี่ยงสูง ({risk_pct:.1f}%)")

    st.progress(float(prob))

    st.subheader("ปัจจัยที่ส่งผลต่อการประเมิน")
    feat_imp = pd.Series(
        model.named_steps['model'].feature_importances_,
        index=input_data.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    feat_imp.plot(kind='barh', ax=ax, color='#534AB7')
    ax.set_title('Feature Importance')
    st.pyplot(fig)