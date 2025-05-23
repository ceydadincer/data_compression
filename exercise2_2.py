import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# print floats as decimals and not in scientific notation
np.set_printoptions(suppress=True)

# load the image as an array of shape (H, W, C)
"""
H (height) and W (weight) are the coordinates of the pixel, C is the channels or color components 
of a pixel the image is an RGB image, so C is 3. Each pixel is represented with an RGB value, 
which is an array of 3 integers, each integer representing the intensity of the corresponding color 
( R -> 0, G -> 1, B -> 2)
"""
img = mpimg.imread('car.jpg')

# normalize pixel values to [0, 1]
img = img / 255.0

# height: the number of pixels vertically
# width: the number of pixels horizontally
# channels: number of channels (features)
height, width, channels = img.shape

# reshape the 3D array to be a 2D array (H*W,C), essentially an array of all pixels with their channels
pixels = img.reshape(height * width, channels)

# an array of the number of clusters/centroids
k_values = [2, 4, 32]

# container for the compressed images and losses for each k value
all_compr_imgs = []
all_losses = []


# function for the random selection of the initial centroids
def initialize_centroids(input_data: np.ndarray, input_k: int) -> np.ndarray:

    # number of data points
    num_points = input_data.shape[0]

    # select input_k many random indices between 0 and num_points
    indices = np.random.choice(num_points, input_k, replace=False)

    # select the rows (input_data points) with the chosen indices as initial centroids
    initial_centroids = input_data[indices]

    # return initialized centroids
    return initial_centroids


# function for the assignment of the closest cluster to each data point
def assign_clusters(input_data: np.ndarray, p_centroids: np.ndarray) -> np.ndarray:

    # calculate the distances of all data points to all centroids
    """ data[:, np.newaxis] gives data a new dimension, it pushes the dimension that
    was initially 2. to 3. then makes the 2. dimension have one element. Essentially turning
    each data point to 1x3 matrices. Now data has the shape (num_points, 1, num_features).
    p_centroids with a shape (k, num_features) is subtracted. The dimensions are matched
    from right to left (right-most dimension -> -1). Because the number of dimensions of the operands
    doesn't match, the missing dimension is added to cluster_centroids, which is the left-most
    dimension ( , k, num_features) -> (1 , k, num_features). Now with broadcasting, each point is
    subtracted from each centroid by matching each data point to each centroid (all combinations
    of data x centroids). And the features get subtracted easily since they're the same
    number. The result has the shape (num_points, k, num_features).
    With np.linalg.norm we calculate the norm of the features, which is the last dimension.
    The result has the shape (num_points, k), each representing the distance between a data_point
    and a centroid."""
    euc_distances = np.linalg.norm(input_data[:, np.newaxis] - p_centroids, axis=2)

    # a column to contain the indices of the closest clusters of each data point
    # for every row, look at each column and determine at which of its columns the value is the min
    updated_labels = np.argmin(euc_distances, axis=1)

    # return the array of the indices of the assigned clusters
    return updated_labels


# function for the updating of the centroids according to their closest data points
def update_centroids(input_data: np.ndarray, p_labels: np.ndarray, old_centroids: np.ndarray) \
        -> np.ndarray:

    # first initialize the new centroids with the old values (copy so that old_centroids isn't updated)
    updated_centroids = old_centroids.copy()

    # number of centroids
    num_centroids = old_centroids.shape[0]

    # assign clusters to each data point
    for i in range(num_centroids):
        # save the data points of the i-th cluster
        """ compare each data point's assigned cluster's index to i and give a boolean array with the
        result for each point. Then make an array of the points that the comparison is true for """
        cluster_points = input_data[p_labels == i]

        # if the cluster has at least one point, change its centroid to the mean of its points
        if len(cluster_points) > 0:
            updated_centroids[i] = np.mean(cluster_points, axis=0)

        # if the cluster has no data points, its centroid stays the same
        else:
            pass

    # return the new centroids
    return updated_centroids


# function for the computation of the loss after each updating of the centroids
def compute_loss(input_data: np.ndarray, p_centroids: np.ndarray, p_labels: np.ndarray)\
        -> np.float64:

    # calculate sum of squared distances between each cluster and its assigned centroid
    computed_loss = np.sum((input_data - p_centroids[p_labels]) ** 2)

    # return the calculated loss
    return computed_loss


# function for the k_means method
def k_means(input_data: np.ndarray, input_k: int) -> tuple[np.ndarray, np.ndarray, list]:

    # an array of indexes of the centroids assigned to each data point
    current_labels = np.zeros(input_data.shape[0])

    # total losses of each iteration of each iteration
    computed_losses = []

    # initialize the first centroids
    current_centroids = initialize_centroids(input_data, input_k)

    # the maximum change across all centroids (set to a number bigger than 10^(-6) initially)
    max_update_change = 1

    # the number of iterations to keep track
    iterations = 0

    # run the algorithm until the centroids converge
    while max_update_change > 10 ** (-6) and iterations < 1000:

        # increment the number of iterations
        iterations += 1

        # assign the data points new clusters
        current_labels = assign_clusters(input_data, current_centroids)

        # recalculate the centroids
        new_centroids = update_centroids(input_data, current_labels, current_centroids)

        # set to the maximum change of the centroids between iterations
        max_update_change = np.max(np.linalg.norm(current_centroids - new_centroids, axis=1))

        # set the centroids to the recalculated centroids
        current_centroids = new_centroids.copy()

        # compute the loss of the current iteration and add it to the list
        computed_losses.append(compute_loss(input_data, current_centroids, current_labels))

    return current_centroids, current_labels, computed_losses


def main():
    # run the algorithm for different k values
    for k in k_values:

        # get results from the algorithm
        centroids, labels, losses = k_means(pixels, k)

        # replace each pixel with its centroid to compress the image
        compressed_pixels = centroids[labels]

        # reshape the array to again 3D (H, W, C) to visualize the image
        compressed_img = compressed_pixels.reshape(height, width, channels)

        # save the compressed image and the loss of the current k value to the list
        all_compr_imgs.append(compressed_img)
        all_losses.append(losses)

    # plot the loss against the iterations
    # create len(k_values) many subplots
    fig, axes = plt.subplots(1, len(k_values), figsize=(12, 4))
    fig.canvas.manager.set_window_title("K-Means Loss Plots")

    # title and plot each graph
    for i in range(len(k_values)):
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("Loss")
        axes[i].set_title(f"K-Means Loss Convergence \nK = {k_values[i]}")
        axes[i].plot(range(len(all_losses[i])), all_losses[i], marker='o', linestyle='-')

    plt.tight_layout()  # adjusts spacing between subplots
    plt.show(block=False)

    # create 2 subplots to show both original and compressed images
    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(12, 4))
    fig.canvas.manager.set_window_title("K-Means Compressed vs Original Image")

    # show original image
    axes[3].imshow(img)
    axes[3].set_title("Original")
    axes[3].axis("off")  # hides the axis lines, ticks, labels

    # show the compressed image
    for i in range(len(k_values)):
        axes[i].imshow(all_compr_imgs[i])
        axes[i].set_title(f"Compressed with K = {k_values[i]}")
        axes[i].axis("off")

    plt.tight_layout()  # adjusts spacing between subplots
    plt.show()


if __name__ == "__main__":
    main()
