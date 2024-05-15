import open3d as o3d
import abc
from point_cloud_playground.utils import calculate_reorientation_matrix
import numpy as np

class PlaneDetector(abc.ABC):
    """
    Abstract base class for plane detection in point clouds.
    """
    def __init__(self, file_path):
        """
        Parameters:
        - file_path (str): Path to the .ply file containing the point cloud data.
        """
        self.point_cloud = o3d.io.read_point_cloud(file_path)

    @abc.abstractmethod
    def detect_plane(self):
        pass

    @abc.abstractmethod
    def detect_and_reorient_plane(self):
        pass

class PlaneDetectorRANSAC(PlaneDetector):
    """
    A class for detecting a plane in a point cloud using RANSAC.
    """
    def __init__(self, file_path, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """
        Initializes the PlaneDetectorRANSAC object.

        Parameters:
        - file_path (str): Path to the .ply file containing the point cloud data.
        - distance_threshold (float): Distance threshold for plane segmentation.
        - ransac_n (int): Number of points to sample for RANSAC.
        - num_iterations (int): Number of RANSAC iterations.
        """
        super().__init__(file_path)
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations


    def detect_plane(self):
        """
        Detects a plane in the point cloud using RANSAC.

        Returns:
        - floor_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        self.point_cloud.estimate_normals()
        plane_model, inliers = self.point_cloud.segment_plane(distance_threshold=self.distance_threshold, ransac_n=self.ransac_n, num_iterations=self.num_iterations)
        [a, b, c, d] = plane_model
        floor_equation = (a, b, c, d) 
        return floor_equation
    
    def detect_and_reorient_plane(self):
        """
        Detects and reorients the plane in the point cloud using RANSAC.

        Returns:
        - floor_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        floor_equation = self.detect_plane()
        transformation_matrix = calculate_reorientation_matrix(floor_equation)
        self.point_cloud.transform(transformation_matrix)

        return floor_equation

class PlaneDetectorPCA(PlaneDetector):
    """
    A class for detecting a plane in a point cloud using Principal Component Analysis (PCA).
    """
    def __init__(self, file_path):
        super().__init__(file_path)

    def detect_plane(self):
        """
        Detects a plane in the point cloud using PCA.

        Returns:
        - floor_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        points = np.asarray(self.point_cloud.points)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        covariance_matrix = np.dot(points_centered.T, points_centered) / len(points)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        d = -np.dot(normal, centroid)
        normal /= np.linalg.norm(normal)
        floor_equation = tuple(normal.tolist() + [d])
        return floor_equation
    
    def detect_and_reorient_plane(self):
        """
        Detects and reorients the plane in the point cloud using PCA.

        Returns:
        - floor_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        floor_equation = self.detect_plane()
        transformation_matrix = calculate_reorientation_matrix(floor_equation)
        self.point_cloud.transform(transformation_matrix)
        return floor_equation


class PlaneDetectorConvexHull(PlaneDetector):
    """
    A class for detecting a plane in a point cloud using Convex Hull method.
    """
    def __init__(self, point_cloud):
        super().__init__(point_cloud)

    def detect_plane(self):
        """
        Detects a plane in the point cloud using Convex Hull method.

        Returns:
        - floor_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        hull, _ = self.point_cloud.compute_convex_hull()
        plane_points = np.asarray(hull.vertices)
        floor_equation = self.fit_plane(plane_points)
        return floor_equation
    
    def detect_and_reorient_plane(self):
        """
        Detects and reorients the plane in the point cloud using Convex Hull method.

        Returns:
        - floor_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        floor_equation = self.detect_plane()
        transformation_matrix = calculate_reorientation_matrix(floor_equation)
        self.point_cloud.transform(transformation_matrix)
        return floor_equation


    def fit_plane(self, points):
        """
        Fits a plane to a set of points using least squares method.

        Parameters:
        - points (numpy.ndarray): Points on the plane.

        Returns:
        - plane_equation (tuple): Coefficients of the plane equation (a, b, c, d) where ax + by + cz + d = 0.
        """
        centroid = np.mean(points, axis=0)
        covariance_matrix = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        normal = eigenvectors[np.argmin(eigenvalues)]
        d = -np.dot(normal, centroid)
        normal /= np.linalg.norm(normal)
        plane_equation = tuple(normal.tolist() + [d])
        return plane_equation
