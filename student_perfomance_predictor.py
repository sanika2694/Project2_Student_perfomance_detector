import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Optional: Create sample data
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 45, 50, 60, 65, 70, 75, 80, 90, 95],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
df.to_csv("student_data.csv", index=False)

# Load dataset
data = pd.read_csv("student_data.csv")


X = data[["Hours_Studied", "Attendance"]]
y = data["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline (scaling + model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Take user input
try:
    h = float(input("\n📚 Enter study hours: "))
    a = float(input("📅 Enter attendance percentage: "))

    user_data = np.array([[h, a]])
    prediction = pipeline.predict(user_data)
    prob = pipeline.predict_proba(user_data)

    print(f"\n🧠 AI Prediction: {'PASS ✅' if prediction[0] == 1 else 'FAIL ❌'}")
    print(f"🔢 Confidence: {round(prob[0][1]*100, 2)}% chance of PASS")

except ValueError:
    print("❌ Invalid input! Please enter numeric values.")

# Bonus: Decision Boundary Plot (2D)
def plot_decision_boundary(X, y, model):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 5, X.iloc[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k')
    plt.xlabel("Hours Studied")
    plt.ylabel("Attendance %")
    plt.title("🧠 Logistic Regression Decision Boundary")
    plt.show()

# Plot decision boundary
plot_decision_boundary(X, y, pipeline)

