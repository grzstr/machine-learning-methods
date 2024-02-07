# Machine learning methods

In this project, proprietary solutions of the k nearest neighbor method and perceptron were implemented. The classification was carried out on a database containing congressional voting results from 1984 from the University of California's UCI database. Also used were ready-made implementations of machine learning algorithms available in the scikit-learn library.

## Description of the dataset
The dataset contains congressional voting results from 1984 and is available in the University of California's UCI database. Each record represents a single Congressional representative, and the attributes describe the vote on various bills. The data is used to perform classification, identifying the political preferences of individual representatives.

## Implemented methods

1. **k-Nearest Neighbors (KNN):**
   - We implemented a custom version of the KNN algorithm, which relies on the distance between a sample and its nearest neighbors in the feature space.
   - The algorithm allows flexible adjustment of the number of neighbors and distance metric.

2. **Perceptron:**
   - Custom implementation of the perceptron, one of the simplest machine learning algorithms for binary classification.
   - The algorithm operates on the principle of learning through iterative adjustment of feature weights.

3. **Utilized Classification Methods from scikit-learn:**
   - The project also leverages pre-built classification methods from the `scikit-learn` library, including KNN, MultiLayer Perceptron, Decision Tree, and Support Vector Classifier (SVC).

## Tools Utilized

The project incorporates both custom implementations and ready-made methods from `scikit-learn`, providing a comprehensive comparison between custom solutions and standard implementations.

