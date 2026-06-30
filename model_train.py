import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# load tarining data
train_df = pd.read_csv("loan_train.csv")

# features and target
features = ["Age","Annual_Income","Credit_Score","Loan_Amount",
            "Employment_Years","Debt_To_Income_Ratio","Dependents"]

X_train = train_df[features]
y_train = train_df["Approved"]  

# Feature Scalling by standscaller

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Knn (k = 7)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled,y_train)

# save model

with open("knn_model.pkl","wb") as f:
    pickle.dump(knn,f)

with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

print("Model and scaler saved: knn_model.pkl, scaler.pkl")