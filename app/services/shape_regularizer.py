"""
Shape Regularizer - Fit contours to ideal geometric shapes based on class.
"""
import cv2
import numpy as np

class ShapeRegularizer:
    """
    Applies shape fitting algorithms to contours to regularize them into
    ideal geometric shapes (e.g., rectangles, hexagons) based on a given class name.
    """

    def regularize(self, contour: np.ndarray, class_name: str) -> np.ndarray:
        """
        Main dispatcher. Calls the appropriate regularization function based on class name.
        """
        if class_name == "slope_flat":
            return self.fit_rectangle(contour)
        elif class_name == "slope_hip":
            return self.fit_hexagon(contour)
        else:
            # For other shapes, just simplify to remove noise
            return self.simplify_polygon(contour)

    def simplify_polygon(self, contour: np.ndarray, factor: float = 0.01) -> np.ndarray:
        """
        Simplifies a polygon using Douglas-Peucker algorithm.
        """
        epsilon = factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def fit_rectangle(self, contour: np.ndarray) -> np.ndarray:
        """
        Fits a contour to the minimum area bounding rectangle.
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        return box.reshape(-1, 1, 2)

    def fit_hexagon(self, contour: np.ndarray) -> np.ndarray:
        """
        Fits a contour to a hexagon by finding the convex hull and simplifying.
        This is a placeholder and can be improved.
        """
        # For now, we use a simple approach: find convex hull and simplify
        hull = cv2.convexHull(contour)
        
        # We could try to find 6 dominant points, but for now, simplify
        return self.simplify_polygon(hull, factor=0.04)