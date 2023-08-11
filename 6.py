import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Ignore the convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters and their values to search
hyperparameters = {
    'hidden_layer_sizes': [(50,), (100,), (200,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Initialize the classifier
classifier = MLPClassifier(random_state=42, max_iter=500)

# Perform grid search using cross-validation
grid_search = GridSearchCV(classifier, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)

# Retrieve the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_hyperparameters = grid_search.best_params_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print("Best Hyperparameters:", best_hyperparameters)
print("Accuracy:", accuracy)
