# IPL Winner Prediction System Using AI & ML

## Introduction

The IPL Winner Prediction System uses Artificial Intelligence (AI) and Machine Learning (ML) techniques to predict the winning team of IPL matches. The system analyzes previous IPL match data such as team performance, player statistics, venue details, toss results, and run rates. Machine learning algorithms process this data and generate predictions with accuracy. This project helps cricket fans, analysts, and researchers understand match outcomes using data science techniques.

---

# Technologies Used

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Main programming language  |
| Jupyter Notebook | Model training and testing |
| Pandas           | Data preprocessing         |
| NumPy            | Numerical operations       |
| Scikit-learn     | ML algorithms              |
| Matplotlib       | Graphs and charts          |
| Streamlit        | Frontend deployment        |
| CSV Dataset      | Historical IPL data        |

---

# Working of the System

## Step 1: Data Collection

Collect IPL datasets containing:

* Team names
* Match venue
* Toss winner
* Runs scored
* Wickets
* Player performance
* Match result

---

## Step 2: Data Preprocessing

Data cleaning operations:

* Remove null values
* Convert categorical data into numerical form
* Feature selection
* Data normalization

Example:

```python
df.dropna(inplace=True)
```

---

## Step 3: Feature Engineering

Important features:

* Batting strength
* Bowling strength
* Win percentage
* Home ground advantage
* Toss impact

---

## Step 4: Model Training

Common ML Algorithms:

* Logistic Regression
* Random Forest
* Decision Tree
* XGBoost

Example:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

## Step 5: Prediction

The trained model predicts:

* Winning probability
* Match winner
* Team performance

Example:

```python
prediction = model.predict(X_test)
```

---

# System Architecture Diagram

```text
+------------------+
| IPL Dataset      |
+------------------+
          |
          v
+------------------+
| Data Preprocessing|
+------------------+
          |
          v
+------------------+
| Feature Selection |
+------------------+
          |
          v
+------------------+
| ML Model Training |
+------------------+
          |
          v
+------------------+
| Prediction Result |
+------------------+
```

---

# Flowchart Diagram

```text
Start
   |
   v
Collect IPL Data
   |
   v
Clean Dataset
   |
   v
Train ML Model
   |
   v
Test Accuracy
   |
   v
Predict Winner
   |
   v
Display Result
   |
   v
End
```

---

<img width="412" height="122" alt="image" src="https://github.com/user-attachments/assets/a4e761a8-b001-44c8-a8d4-3d1ed3d3c226" />


---

# Sample Prediction Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("ipl.csv")

# Features and Target
X = df[['team1', 'team2', 'toss_winner']]
y = df['winner']

# Convert categorical data
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
```

---

# Advantages

* Fast prediction results
* Accurate analysis using ML
* Helps cricket analysts
* Real-time data processing
* Improves decision-making

---

# Future Enhancements

* Live score prediction
* Deep learning integration
* Player-level prediction
* Mobile application development
* Real-time API integration

<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/467cd761-edc3-4da1-bfd1-0e1a3018b588" />





Toss dicision analyse



<img width="1740" height="515" alt="image" src="https://github.com/user-attachments/assets/1fc5615b-afb8-4f7c-9d6e-39a0bdfb546a" />




Venue imapact analysis



<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/b963739d-5afe-4c09-97f3-9066ed8b0cbb" />




Tam performance coparision

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/54e1f911-6905-4ce9-b51d-441ea5dc169f" />




Score win margin distibution

<img width="1590" height="617" alt="image" src="https://github.com/user-attachments/assets/2a3bd9b4-4e2b-4366-bb1f-381c9e83811a" />





Feature core-relation heat map

<img width="1417" height="1298" alt="image" src="https://github.com/user-attachments/assets/24adb9e4-6652-424b-a8ff-3016807b2fcc" />







season win prediction

<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/1fc07b27-fac2-4123-8405-5be31392a59e" />





confusion metrix visulation

<img width="2389" height="515" alt="image" src="https://github.com/user-attachments/assets/81fc5f62-3421-40e8-8d15-f2bd449c2aa9" />





ROC CURVE

<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/b043b0d9-3159-4b49-b55e-fa521b0fbdf9" />







XG BOOST


<img width="1187" height="990" alt="image" src="https://github.com/user-attachments/assets/32b495e2-1f88-47ea-8d15-d2c1455ac326" />

---

# Conclusion

The IPL Winner Prediction System using AI & ML is an intelligent cricket analytics project that predicts match outcomes using historical IPL data and machine learning algorithms. It improves prediction accuracy through data preprocessing, feature engineering, and model training techniques. The project demonstrates the practical implementation of AI and ML in sports analytics.






https://www.kaggle.com/code/shadab80k/ipl-2026-winner-prediction-model?scriptVersionId=309101702&cellId=30

https://www.kaggle.com/code/shadab80k/ipl-2026-winner-prediction-model?scriptVersionId=309101702&cellId=10
