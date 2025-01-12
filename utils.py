import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def load_ply_file(file_path, normalize=True):
    """ read a point cloud from a .ply file and normalize it

    Args:
        file_path (str): path to ply file
        normalize (bool, optional): if set to true, normalizes the ply file. Defaults to True.

    Returns:
        o3d object: cloud object
    """  

    cloud = o3d.io.read_point_cloud(file_path)
    if normalize:
        cloud.scale(1.0 / np.max(cloud.get_max_bound() - cloud.get_min_bound()), center=cloud.get_center())
    return cloud


def plot_2d_o3d(cloud, title="", ):
    """ plot a 2D projection of a point cloud

    Args:
        cloud (o3d object): point cloud
        title (str, optional): title of the plot. Defaults to "".
    """    
    points = np.asarray(cloud.points)
    fig = go.Figure(data=[go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers')])
    fig.update_layout(title=title)
    fig.show()


def plot_mat(points, nose_tip=None, save_path=None):
    projected_points = points[:, :2]  # Keep only the x and y coordinates

    # Create a 2D image (a scatter plot in this case)
    plt.figure(figsize=(8, 8))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=3, marker='.')

    # Set labels and title
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("2D Projection of Point Cloud")

    if nose_tip is not None:
        plt.scatter(nose_tip[0], nose_tip[1], c='r', marker='o')
    if save_path is not None:
        plt.savefig(save_path)

    # Display the image
    plt.show()

def draw_geometries(geometries, points_to_draw):
    """ create a 3D plot of the point clouds using plotly

    Args:
        geometries (list of o3d cload objects): list of point clouds
        points_to_draw (list of points which are lists with 3 elements): points to draw
    """    

    graph_objects = []

    for geometry in geometries:
        points = np.asarray(geometry.points)

        # defineing colors
        colors = None
        if geometry.has_colors():
            colors = np.asarray(geometry.colors)
        elif geometry.has_normals():
            colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
        else:
            geometry.paint_uniform_color((1.0, 0.0, 0.0))
            colors = np.asarray(geometry.colors)

        # plot the point cloud
        scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
        graph_objects.append(scatter_3d)

    # plot the points
    for point in points_to_draw:
        nose_tip = point
        nose_tip_3d = go.Scatter3d(x=[nose_tip[0]], y=[nose_tip[1]], z=[nose_tip[2]], mode='markers', marker=dict(size=2, color='blue'))
        graph_objects.append(nose_tip_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )

    fig.show()


def plot_3d_o3d(cloud, nose_tip=None, save_path=None):
    """ plot a 3D point cloud using matplotlib

    Args:
        cloud (o3d object): point cloud
        nose_tip (np.array, optional): nose tip. Defaults to None.
        save_path (str, optional): path to save the plot. Defaults to None.
    """    

    points = np.asarray(cloud.points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    if nose_tip is not None:
        ax.scatter(nose_tip[0], nose_tip[1], nose_tip[2], c='r', marker='o')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def get_most_protruding_point(cloud):
    """ get the most protruding point of a point cloud

    Args:
        cloud (o3d object): point cloud

    Returns:
        np.array: most protruding point
    """    

    points = np.asarray(cloud.points)
    max_z_index = np.argmax(points[:, 2])
    return points[max_z_index]


def DBSCAN_clustering(cloud, eps=0.024, min_samples=21):
    """ cluster a point cloud using DBSCAN

    Args:
        cloud (o3d object): point cloud
        eps (float, optional): epsilon parameter of DBSCAN. Defaults to 0.02.
        min_samples (int, optional): min_samples parameter of DBSCAN. Defaults to 10.

    Returns:
        list of np.arrays: list of clusters
    """    

    points = np.asarray(cloud.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    clusters = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster = points[labels == label]
        clusters.append(cluster)
    return clusters


def find_nose_cluster(cloud, clusters):
    """ find the cluster that contains the most protruding point

    Args:
        cloud (o3d object): point cloud
        clusters (list of np.arrays): list of clusters

    Returns:
        np.array: nose cluster
    """    

    most_protruding_point = get_most_protruding_point(cloud)
    for cluster in clusters:
        if most_protruding_point in cluster:
            return cluster
    return None


def find_middle_point(cluster):
    """ find the X-axis middle point of a cluster

    Args:
        cluster (np.array): cluster

    Returns:
        int: middle point X-axis
    """

    min_x = np.min(cluster[:, 0])
    max_x = np.max(cluster[:, 0])    

    return (min_x + max_x) / 2