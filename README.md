# 🤖 Machine Learning Project: Implementing a Neural Network for Binary Classification

In this project, you will build, train, and evaluate a **feedforward neural network** model using **Keras** to solve a binary classification problem. You will experiment with different hyperparameters and neural network configurations to improve model performance.

---

### 📌 Project Tasks

* Load the Airbnb "listings" dataset
* Define the prediction label and identify features
* Prepare and clean the data for modeling
* Create labeled examples
* Split the data into training and test sets
* Construct a feedforward neural network model using Keras
* Train the neural network on the training data
* Evaluate model performance on training, validation, and test datasets
* Experiment with hyperparameters and architecture to optimize the model

---

### 🛠️ Example Code Snippet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
airbnb_df = pd.read_csv('airbnb_listings.csv')

# Define label and features
label = 'some_binary_label'  # replace with actual label column name
features = ['feature1', 'feature2', 'feature3']  # replace with actual feature columns

X = airbnb_df[features]
y = airbnb_df[label]

# Data preparation: split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test accuracy: {accuracy:.4f}')
```
