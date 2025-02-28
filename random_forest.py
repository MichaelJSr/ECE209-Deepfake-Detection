import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from images_analysis import (
    FAKE_EMBEDDINGS_DIR,
    REAL_EMBEDDINGS_DIR,
    load_embeddings,
)


# Load training data
print("\nLoading training data...")
fake_train_embeddings = load_embeddings(FAKE_EMBEDDINGS_DIR)
fake_train_labels = np.zeros(len(fake_train_embeddings))
real_train_embeddings = load_embeddings(REAL_EMBEDDINGS_DIR)
real_train_labels = np.ones(len(real_train_embeddings))

# Combine training data
train_embeddings = np.vstack((fake_train_embeddings, real_train_embeddings))
train_labels = np.concatenate((fake_train_labels, real_train_labels))

# Load test data
fake_test_embeddings = load_embeddings(FAKE_EMBEDDINGS_DIR)
fake_test_labels = np.zeros(len(fake_test_embeddings))
real_test_embeddings = load_embeddings(REAL_EMBEDDINGS_DIR)
real_test_labels = np.ones(len(real_test_embeddings))

# Combine test data
test_embeddings = np.vstack((fake_test_embeddings, real_test_embeddings))
test_labels = np.concatenate((fake_test_labels, real_test_labels))

# Train Random Forest model
print("Training Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=None,      # Maximum depth of trees (None means unlimited)
    min_samples_split=2, # Minimum samples required to split a node
    min_samples_leaf=1,  # Minimum samples at a leaf node
    max_features='sqrt', # Number of features to consider for best split
    n_jobs=-1,           # Use all available cores
    random_state=42      # For reproducibility
)

# Fit the model
rf_model.fit(train_embeddings, train_labels)

# Predict on test set
predictions = rf_model.predict(test_embeddings)
probabilities = rf_model.predict_proba(test_embeddings)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print(classification_report(test_labels, predictions, target_names=['Fake', 'Real']))

# Create confusion matrix
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to 'confusion_matrix.png'")

# Feature importance analysis
feature_importances = rf_model.feature_importances_
top_indices = np.argsort(feature_importances)[-20:]  # Top 20 features
top_importances = feature_importances[top_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_indices)), top_importances, align='center')
plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved to 'feature_importance.png'")

# Save the model
import joblib
joblib.dump(rf_model, 'deepfake_detector_rf.joblib')
print("Model saved as 'deepfake_detector_rf.joblib'")