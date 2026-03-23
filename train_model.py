import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/cs-training.csv', index_col=0)
df['age'] = df['age'].replace(0, np.nan)

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        n_estimators=200,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'models/best_model.pkl')
print("Train และบันทึกโมเดลเรียบร้อย!")