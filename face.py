import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sklearn.metrics import silhouette_score
from collections import Counter




# Define the image folder path
image_folder = "images"  # Replace with your actual folder path

# Check if the folder exists
if not os.path.exists(image_folder):
    print(f"Error: The folder '{image_folder}' does not exist.")
    exit(1)

# Read images from the folder
images = []
dimension = set()
labels= []
for label in os.listdir(image_folder): 
    new_path = os.path.join(image_folder, label)
    labels.append(label)
    for filename in os.listdir(new_path):  # Correct usage of os.listdir()
        img_path = os.path.join(new_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            dimension.add(img.shape)
        else:
            print(f"Could not read image: {filename}")

print(f"Total images loaded: {len(images)}")
print(f"dimension of images: {(dimension)}")

# Visualize the images
def visualize_images(images):
    plt.figure(figsize=(12, 6))
    for i,img in enumerate(images[:5]):
        try:
           
            plt.subplot(2, 3, i + 1)  # Create a grid of 2 rows and 3 columns
            plt.imshow(img)
            plt.title(f"{label[i]}")
            plt.axis("off")
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
    plt.tight_layout()
    plt.show()
visualize_images(images)
# Initialize a list to store the flattened image data
flattened_data = []

# Loop through the images, resize, and flatten
for image in images:
    image_array = np.array(image).flatten()  # Flatten the image into a 1D array
    flattened_data.append(image_array)

# Convert the list of flattened images into a NumPy array
data_matrix = np.array(flattened_data)

scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_matrix)

print("Shape of data matrix:", data_matrix.shape)
print("Shape of standardized data:", standardized_data.shape)

pca = PCA()
pca_data = pca.fit_transform(standardized_data)

# Plot the explained variance ratio
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

n_components = sum(explained_variance.cumsum() <= 0.95)  # Adjust the threshold as needed
print(f"Number of components to retain 95% variance: {n_components}")

pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(standardized_data)  # Shape: (num_samples, n_components)

print(f"Reduced data shape: {reduced_data.shape}")

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, c=reduced_data[:, 0], cmap='viridis')
plt.title('PCA Reduced Data Visualization (First 2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Principal Component 1')
plt.show()


# Instantiate HDBSCAN with chosen parameters
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_epsilon=0.5)

# Fit and predict clusters
labels = clusterer.fit_predict(reduced_data)


# Fit HDBSCAN to the reduced data
cluster_labels = clusterer.fit_predict(reduced_data)

# Display the resulting labels
print(f"Cluster labels: {cluster_labels}")

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")


# Scatter plot of the first two PCA components
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=10)
plt.colorbar(label='Cluster Label')
plt.title('HDBSCAN Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Testing different configurations
for min_cluster_size in [5, 10, 15]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(reduced_data)
    print(f"Clusters with min_cluster_size={min_cluster_size}: {len(set(labels)) - (1 if -1 in labels else 0)}")

valid_indices = cluster_labels != -1  # Exclude noise points
if valid_indices.sum() > 1:
    score = silhouette_score(reduced_data[valid_indices], cluster_labels[valid_indices])
    print(f"Silhouette Score: {score}")
else:
    print("Not enough valid clusters for silhouette scoring.")

noise_level = 0.1  # Adjust this value to control noise intensity
noisy_data = reduced_data + np.random.normal(0, noise_level, reduced_data.shape)

plt.scatter(noisy_data[:, 0], noisy_data[:, 1], alpha=0.5, s=10, c='grey')
plt.title('PCA Reduced Data with Added Noise')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Apply HDBSCAN to noisy data
noisy_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
noisy_labels = noisy_clusterer.fit_predict(noisy_data)

# Visualize clusters for noisy data
plt.scatter(noisy_data[:, 0], noisy_data[:, 1], c=noisy_labels, cmap='viridis', s=10)
plt.title('HDBSCAN Clustering on Noisy Data')
plt.colorbar(label='Cluster Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Count clusters and noise points
n_clusters_noisy = len(set(noisy_labels)) - (1 if -1 in noisy_labels else 0)
n_noise_noisy = list(noisy_labels).count(-1)
print(f"Clusters on noisy data: {n_clusters_noisy}, Noise points: {n_noise_noisy}")

metrics = ['euclidean', 'manhattan', 'cosine']

for metric in metrics:
    print(f"Testing HDBSCAN with metric: {metric}")
    
    # Apply HDBSCAN with the current metric
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric=metric)
    labels = clusterer.fit_predict(reduced_data)
    
    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
    
    # Visualize results
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(f'HDBSCAN Clustering with {metric.capitalize()} Distance')
    plt.colorbar(label='Cluster Label')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Evaluate clustering quality (ignoring noise points)
valid_indices = noisy_labels != -1
if valid_indices.sum() > 1:
    score = silhouette_score(noisy_data[valid_indices], noisy_labels[valid_indices])
    print(f"Silhouette Score on noisy data: {score}")
else:
    print("Not enough valid clusters for silhouette scoring.")
  

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=10)
plt.colorbar(label='Cluster Label')
plt.title('HDBSCAN Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Count cluster sizes (excluding noise)
cluster_counts = Counter(cluster_labels[~is_noise])

print("Cluster Size Distribution:")
for cluster_id, size in sorted(cluster_counts.items()):
    print(f"Cluster {cluster_id}: {size} points")

# Plot distribution
plt.figure(figsize=(8, 6))
plt.bar(cluster_counts.keys(), cluster_counts.values(), color='skyblue')
plt.title('Cluster Size Distribution')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Points')
plt.grid()
plt.show()

# Load and preprocess new image
new_image = cv2.imread("path_to_new_image.jpg").flatten()  # Replace with the actual path
new_image_std = scaler.transform([new_image])  # Standardize
new_image_pca = pca.transform(new_image_std)  # Reduce dimensions

# Compute distances to existing clusters' exemplars
distances = []
for exemplar in clusterer.exemplars_:
    cluster_center = np.mean(exemplar, axis=0)  # Approximate cluster center
    distance = np.linalg.norm(new_image_pca - cluster_center)
    distances.append(distance)

# Assign the new image to the closest cluster
assigned_cluster = np.argmin(distances)
print(f"Assigned Cluster: {assigned_cluster}")

representative_images = []

# Iterate over each cluster (excluding noise)
for cluster_id in set(cluster_labels):
    if cluster_id == -1:  # Skip noise
        continue
    
    # Get indices of points in the current cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    
    # Compute cluster center
    cluster_points = reduced_data[cluster_indices]
    cluster_center = np.mean(cluster_points, axis=0)
    
    # Find the point closest to the center
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    closest_point_idx = cluster_indices[np.argmin(distances)]
    
    # Save the representative image
    representative_images.append(closest_point_idx)

    print(f"Cluster {cluster_id}: Representative Image Index: {closest_point_idx}")
# Display representative images
for idx in representative_images:
    plt.imshow(images[idx].reshape(original_shape), cmap='gray')  # Adjust for your image format
    plt.title(f"Representative Image for Cluster {cluster_labels[idx]}")
    plt.show()


