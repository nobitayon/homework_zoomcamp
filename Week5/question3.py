import pickle
from flask import Flask

model_file='model1.bin'

with open(model_file,'rb') as f_in:
    model=pickle.load(f_in)

dv_file='dv.bin'
with open(dv_file,'rb') as f_in:
    dv=pickle.load(f_in)

customer={"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X=dv.transform([customer])
y_pred=model.predict_proba(X)[0,1]
churn=y_pred>=0.5

result={
    "churn_probability":y_pred,
    "churn":churn
}
print(result)