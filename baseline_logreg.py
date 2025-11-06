from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from baseline_features import load_dataset

# Load dataset
train_dir = 'pokemon_split/train'
val_dir = 'pokemon_split/val'

print("Extracting training features...")
X_train, y_train, class_names = load_dataset(train_dir)

print("Extracting validation features...")
X_val, y_val, _ = load_dataset(val_dir)

# Train baseline logistic regression
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, verbose=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# Optionally save model
joblib.dump(model, "baseline_logreg.pkl")
