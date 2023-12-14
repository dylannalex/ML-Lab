import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Image:
    def __init__(self, rgb_matrix: np.ndarray, title: str = None):
        self.rgb_matrix = rgb_matrix
        self.rgb_vector = rgb_matrix.reshape(-1, 3)
        self.shape = rgb_matrix.shape
        self.title = title


def load_image(image_path: str) -> Image:
    rgb_matrix = mpimg.imread(image_path)
    return Image(rgb_matrix)


def rgb_vector_to_image(rgb_vector: np.ndarray, image_shape: tuple):
    return Image(rgb_vector.reshape(image_shape))


def plot_rgb_vector(
    rgb_vector: np.ndarray,
    figsize=(10, 5),
    title: str = None,
    alpha: float = 0.6,
    s: int = 4,
):
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(projection="3d")
    r, g, b = rgb_vector[:, 0], rgb_vector[:, 1], rgb_vector[:, 2]
    colors = rgb_vector / 255
    axes.scatter(r, g, b, c=colors, marker="o", s=s, alpha=alpha)
    axes.set_title(title)
    plt.tight_layout()
    plt.show()


def display_image(image: Image, figsize=(10, 5)):
    fig, axes = plt.subplots(figsize=figsize)
    axes.imshow(image.rgb_matrix)
    axes.set_title(image.title)
    axes.axis("off")
    plt.show()


def display_images(*images: Image, figsize=(10, 5)):
    # One image
    if len(images) == 1:
        display_image(*images, figsize)
        return

    # Multiple images
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for i, image in enumerate(images):
        axes[i].imshow(image.rgb_matrix)
        axes[i].set_title(image.title)
        axes[i].axis("off")
    plt.show()
