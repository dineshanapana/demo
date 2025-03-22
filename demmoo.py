import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.mean = {}
        self.variance = {}
        self.prior = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            # Select instances belonging to class c
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.prior[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Calculate the posterior probability for class c
                prior = np.log(self.prior[c])
                likelihood = self.calculate_likelihood(x, c)
                posterior = prior + likelihood
                posteriors.append(posterior)
            # Choose the class with the highest posterior probability
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

    def calculate_likelihood(self, x, c):
        mean = self.mean[c]
        variance = self.variance[c]
        # Use Gaussian probability density function
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / (variance + 1e-6))
        likelihood = np.prod(exponent / np.sqrt(2 * np.pi * variance + 1e-6))
        return np.log(likelihood)  # Return the log of the likelihood

# Example usage
if __name__ == "__main__":
    # Generate a simple dataset
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Naïve Bayes model
    model = NaiveBayes()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
