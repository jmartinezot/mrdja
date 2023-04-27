from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.spatial import procrustes
from typing import List

class ShapeClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses Procrustes analysis to classify 3D shapes based on their similarity to a set of medoids.

    :param medoids: A list of medoid shapes to compare against.
    :type medoids: List[np.ndarray]
    """
    def __init__(self, medoids):
        self.medoids = medoids

    def fit(self, X):
        """
        Fit the classifier to a set of training shapes.

        :param X: The set of training shapes to fit the classifier to.
        :type X: List[np.ndarray]
        :return: The fitted classifier object.
        :rtype: ShapeClassifier
        """
        self.medoids = X
        return self

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict the class labels of a set of test shapes.

        :param X: The set of test shapes to predict class labels for.
        :type X: List[np.ndarray]
        :return: An array of predicted class labels.
        :rtype: np.ndarray
        """
        return np.array([ShapeClassifier.predict_class(self.medoids, shape) for shape in X])

    def predict_proba(self, X: List[np.ndarray], use_weighted: bool = False) -> np.ndarray:
        """
        Predict the class probabilities of a set of test shapes.

        :param X: The set of test shapes to predict class probabilities for.
        :type X: List[np.ndarray]
        :param use_weighted: Whether to use weighted probabilities based on distance, defaults to False.
        :type use_weighted: bool
        :return: An array of predicted class probabilities.
        :rtype: np.ndarray
        """
        return np.array([ShapeClassifier.predict_class(self.medoids, shape, soft=True, use_weighted=use_weighted) for shape in X])
        
    @staticmethod
    def procrustes_disparity(shape_1: np.ndarray, shape_2: np.ndarray) -> float:
        """
        Calculate the Procrustes disparity between two shapes.

        :param shape_1: The first shape.
        :type shape_1: np.ndarray
        :param shape_2: The second shape.
        :type shape_2: np.ndarray
        :return: The Procrustes disparity between the two shapes.
        :rtype: float
        """
        return procrustes(shape_1, shape_2)[2]

    def predict_class(medoids: np.ndarray, shape: np.ndarray, soft: bool = False, use_weighted: bool = False) -> np.ndarray:
        """
        Predicts the class of a given shape by comparing it to a set of medoids using the Procrustes distance.

        :param medoids: Array of shape medoids used to compare the input shape.
        :type medoids: np.ndarray
        :param shape: Input shape to classify.
        :type shape: np.ndarray
        :param soft: If True, returns a soft classification, where the output is an array of probabilities of belonging to each class. If False, returns a hard classification, where the output is the index of the closest medoid.
        :type soft: bool, optional
        :param use_weighted: If True and soft=True, uses weighted distances instead of the exponential function to calculate the probabilities.
        :type use_weighted: bool, optional
        :return: If soft=True, returns an array of probabilities of belonging to each class. If soft=False, returns the index of the closest medoid.
        :rtype: np.ndarray
        """
        dists = np.array([ShapeClassifier.procrustes_disparity(medoid, shape) for medoid in medoids])
        if soft:
            min_dist = np.min(dists)
            if min_dist == 0:
                # If the minimum distance is zero, return a one-hot encoding with a 1 at the minimum distance index
                proba = np.zeros(len(medoids))
                proba[np.argmin(dists)] = 1
            else:
                # Otherwise, calculate the probabilities based on the distances and normalize them
                if use_weighted:
                    proba = 1.0 / dists
                else:
                    proba = np.exp(-dists / min_dist)
                proba /= np.sum(proba)
            return proba
        else:
            # Return the index of the closest medoid
            return np.argmin(dists)