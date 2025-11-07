from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from baseline_features import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_dir = 'pokemon_split/train'
val_dir = 'pokemon_split/val'

print("Extracting training features...")
X_train, y_train, class_names = load_dataset(train_dir)

print("Extracting validation features...")
X_val, y_val, _ = load_dataset(val_dir)

print("Training k-NN model...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# Compute full confusion matrix
cm_full = confusion_matrix(y_val, y_pred)

misclassified_counts = cm_full.sum(axis=1) - np.diag(cm_full)
top50_indices = np.argsort(misclassified_counts)[-50:][::-1] 

# Subset the confusion matrix
cm_subset = cm_full[np.ix_(top50_indices, top50_indices)]
subset_class_names = [class_names[i] for i in top50_indices]

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
            xticklabels=subset_class_names, yticklabels=subset_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Top 50 Most Misclassified Classes\n (k-NN)')
plt.show()
