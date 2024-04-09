import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gvf.GVF2D import GVF


def get_angles_from_gvf(edge_map, mu=0.2, iter=1000, tol=0.01):
    u, v, converged = GVF(edge_map, mu, iter, tol)
    if converged:
        print('converged')
    angles = np.arctan2(v, u)
    return angles, u, v


def plot_gvf_and_angles(angles, u, v, edge_map):
    Y, X = np.mgrid[0:angles.shape[0], 0:angles.shape[1]]

    # Plot streamlines using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, v, u)  # Note: The order of `u` and `v` is reversed for `streamplot`
    plt.imshow(angles, cmap='hsv')
    # plt.imshow(edge_map, cmap='gray', alpha=0.5)  # Optionally overlay the original image
    plt.colorbar(label='Angle (radians)')
    plt.title('GVF Streamlines')
    plt.axis('off')  # Hide the axis
    plt.show()


if __name__ == "__main__":
    edge_map = np.array(Image.open("UShape.png"))[:, :, 0] / 255
    angles, u, v = get_angles_from_gvf(edge_map)

    plot_gvf_and_angles(angles, u, v, edge_map)
