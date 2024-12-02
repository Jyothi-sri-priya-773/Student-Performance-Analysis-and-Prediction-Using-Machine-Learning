
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from Kaggle source
url = "https://www.kaggleusercontent.com/spscientist/students-performance"
data = pd.read_csv(url)

# Display dataset information
print(data.info())

# Rename columns for easier handling
data.columns = [
    "gender",
    "ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "math_score",
    "reading_score",
    "writing_score"
]

# Combine test scores into an average score column
data["average_score"] = data[["math_score", "reading_score", "writing_score"]].mean(axis=1)

# Encode categorical variables
label_encoders = {}
categorical_cols = ["gender", "ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature selection
X = data.drop(columns=["math_score", "reading_score", "writing_score", "average_score"])
y = data["average_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Feature importance visualization
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

sns.barplot(x="Importance", y="Feature", data=feature_importances)
plt.title("Feature Importance")
plt.show()
