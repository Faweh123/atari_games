# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import psutil

# Function to monitor memory usage (optional)
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 3):.2f} GB")

# Load dataset with error handling
dataset_path = 'my_paypal_creditcard.csv'
try:
    data_frame = pd.read_csv(dataset_path)
    print("Successfully loaded dataset")
except FileNotFoundError:
    print(f"Error: File '{dataset_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Optimize data types to reduce memory usage
for col in data_frame.select_dtypes(include=['float64']).columns:
    data_frame[col] = data_frame[col].astype('float32')

for col in data_frame.select_dtypes(include=['int64']).columns:
    data_frame[col] = data_frame[col].astype('int32')

# Cleaning dataset
dataset_null = data_frame.isnull().sum()
print("Checking for missing values:")
print(dataset_null)

# Handle missing values in the target variable 'Class'
if data_frame['Class'].isnull().any():
    print("Missing values found in 'Class' column. Dropping rows with NaN values.")
    data_frame = data_frame.dropna(subset=['Class'])  # Drop rows where 'Class' is NaN

# After dropping NaN values, check again for missing values
dataset_null = data_frame.isnull().sum()
print("Checking for missing values after cleaning:")
print(dataset_null)

# Summarize the dataset
dataset_overview = data_frame.describe()
print("\nDataset Overview:")
print(dataset_overview)

# Class Distribution
class_distribution = data_frame['Class'].value_counts()
print("\nClass Distribution:")
print(class_distribution)

# Data Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='viridis')
plt.title('Class Distribution', fontsize=14)
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data_frame.corr(), annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Market Size Evaluation and Calculate percentage distribution of each class
market_size_percent = (class_distribution / class_distribution.sum()) * 100
print("\nMarket Size in Percentage:")
print(market_size_percent)

# Visualize market size distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=market_size_percent.index, y=market_size_percent.values, palette='magma')
plt.title('Market Size Distribution (%)', fontsize=14)
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.show()

# Additional Analysis: Feature Importance
X = data_frame.drop('Class', axis=1)
y = data_frame['Class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Imbalance using SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Reduce synthetic samples
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Feature Importance Visualization
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df[:15], palette='Blues_d')  # Top 15 features
plt.title('Top 15 Feature Importances', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}', color='darkorange')
plt.title('Precision-Recall Curve', fontsize=14)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(fontsize=12)
plt.show()

print(f"\nArea Under Precision-Recall Curve (AUPRC): {auprc:.2f}")

# Save Model
joblib.dump(model, 'credit_card_fraud_model.pkl')
print("\nModel saved as 'credit_card_fraud_model.pkl'")