# Decision Tree Classifier

This project demonstrates the implementation of a Decision Tree classifier from scratch in Python. The algorithm is applied to two datasets: **Wine** and **Iris**, achieving high accuracy.

## Key Skills Demonstrated

- **Object-Oriented Programming (OOP):**  
  Structured using `Node` and `DecisionTree` classes for modularity and readability.
  
- **Algorithm Implementation:**  
  Includes splitting criteria (information gain), entropy calculation, and recursive tree growth.

- **Data Handling:**  
  Works with the Iris and Wine datasets, showcasing adaptability to different datasets.

- **Model Evaluation:**  
  Achieves **94.44%** accuracy on the Wine dataset and **100%** on the Iris dataset.

- **Random Feature Selection:**  
  Incorporates randomness in feature selection for better generalization.

## Results

| Dataset | Accuracy |
|---------|----------|
| Wine    | 94.44%   |
| Iris    | 100%     |

### Analysis:
- The Iris dataset has fewer classes (3), making it easier for the Decision Tree to classify accurately.
- The Wine dataset has more classes (13), resulting in slightly lower but still strong accuracy.

## How to Use

1. Install dependencies:
   ```bash
    pip install numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
2. Load your dataset, split it into features (X) and labels (y), and initialize the DecisionTree::
   ```bash  
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and predict
    dt = DecisionTree(max_depth=10)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
