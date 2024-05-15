import open3d as o3d

class PointCloudSmoother:
    """
    A class for smoothing point clouds using different methods.
    """

    def __init__(self, file_path):
        """
        Parameters:
        - file_path (str): Path to the .ply file containing the point cloud data.
        """
        self.point_cloud = o3d.io.read_point_cloud(file_path)
    
    def poisson_surface_reconstruction(self):
        """
        Returns:
        - mesh (open3d.geometry.TriangleMesh): Reconstructed surface mesh.
        """
        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud)
        return mesh

    def moving_least_squares(self, radius):
        """
        Parameters:
        - radius (float): Radius of the local neighborhood for smoothing.

        Returns:
        - smoothed_point_cloud (open3d.geometry.PointCloud): Smoothed point cloud.
        """
        # Moving Least Squares smoothing
        self.point_cloud_smoothed = self.point_cloud.filter_smooth_simple(radius=radius)
        return self.point_cloud_smoothed

    def gaussian_filtering(self, sigma):
        """
        Parameters:
        - sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
        - smoothed_point_cloud (open3d.geometry.PointCloud): Smoothed point cloud.
        """
        # Gaussian filtering
        self.point_cloud_smoothed = self.point_cloud.filter_smooth_gaussian(sigma=sigma)
        return self.point_cloud_smoothed

    def bilateral_filtering(self, sigma_color, sigma_distance):
        """
        Parameters:
        - sigma_color (float): Standard deviation of the Gaussian kernel in the color domain.
        - sigma_distance (float): Standard deviation of the Gaussian kernel in the spatial domain.

        Returns:
        - smoothed_point_cloud (open3d.geometry.PointCloud): Smoothed point cloud.
        """
        # Bilateral filtering
        self.point_cloud_smoothed = self.point_cloud.filter_smooth_bilateral(sigma_color=sigma_color, sigma_distance=sigma_distance)
        return self.point_cloud_smoothed

def main():
    # Load point cloud data
    file_path = "data/shoe_pc.ply"

    # Create PointCloudSmoother object
    smoother = PointCloudSmoother(file_path)

    # Poisson surface reconstruction
    mesh_poisson = smoother.poisson_surface_reconstruction()

    # Moving Least Squares smoothing
    point_cloud_mls = smoother.moving_least_squares(radius=0.02)

    # Gaussian filtering
    point_cloud_gaussian = smoother.gaussian_filtering(sigma=0.02)

    # Bilateral filtering
    point_cloud_bilateral = smoother.bilateral_filtering(sigma_color=0.02, sigma_distance=0.02)

    # Visualize results
    o3d.visualization.draw_geometries([mesh_poisson, point_cloud_mls, point_cloud_gaussian, point_cloud_bilateral])

if __name__ == "__main__":
    main()