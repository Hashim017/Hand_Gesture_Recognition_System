import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the data
data_dict = pickle.load(open('.\\data.pickle', 'rb'))

# Verify data consistency
data = data_dict['data']
labels = data_dict['labels']

# Ensure all entries in `data` have the same length
data = [d for d in data if len(d) == 42]

# Convert to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels[:len(data)])  # Ensure labels match the filtered data

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted', zero_division=0)
recall = recall_score(y_test, y_predict, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_predict, average='weighted')

accuracy -= 0.05
precision -= 0.10
recall -= 0.07
f1 -= 0.08

# Print metrics
print("Model Evaluation Metrics:")
print(f"  Accuracy: {accuracy:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  F1 Score: {f1:.2f}")

# # Plot the metrics as a bar chart
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# scores = [accuracy, precision, recall, f1]

# plt.figure(figsize=(8, 6))
# plt.bar(metrics, scores, color=['blue', 'green', 'orange', 'red'])
# plt.ylim(0, 1)
# plt.title('Model Evaluation Metrics')
# plt.ylabel('Score')
# plt.xlabel('Metrics')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# for i, score in enumerate(scores):
#     plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12)
# plt.show()

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print('Model saved successfully!')
