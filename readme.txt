This project applies the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) algorithm to facial recognition data. 
It aims to cluster facial images in an unsupervised learning scenario, tackling challenges like noise, unbalanced data, and dimensionality reduction. 
The methodology involves preprocessing image data, dimensionality reduction using PCA, 
and clustering with HDBSCAN, followed by visualization and evaluation.
Prerequisites

Python 3.x installed.

Required Python libraries:

numpy

pandas

matplotlib

opencv-python

scikit-learn

hdbscan

Familiarity with:

Python programming

Unsupervised learning and clustering

Dimensionality reduction techniques (e.g., PCA)

Dataset

Use a facial image dataset, such as CelebA or Labeled Faces in the Wild (LFW).

Ensure the dataset contains images in a consistent format.

Steps to Run the Project

1. Setup

Install required libraries:

pip install numpy pandas matplotlib opencv-python scikit-learn hdbscan

Place your dataset in a folder and note the folder path.

2. Data Preprocessing

Load image data using cv2.imread().

Flatten images into 1D arrays and standardize the data using StandardScaler.

3. Dimensionality Reduction

Apply PCA to reduce feature dimensions while retaining essential variance.

Choose the number of components based on cumulative explained variance.

4. Clustering with HDBSCAN

Configure HDBSCAN parameters (e.g., min_cluster_size, metric) and apply to PCA-reduced data.

Extract cluster labels and identify noise points.

5. Visualization and Analysis

Create scatter plots to visualize clusters.

Highlight noise points and analyze their distribution.

Evaluate clustering quality using metrics like silhouette score.

6. Test on New Data

Preprocess a new image similarly to the training data.

Assign the new image to the closest cluster using cluster exemplars.

7. Identify Representative Images

Compute the mean of points in each cluster (cluster center).

Identify the image closest to the cluster center as the representative image.

Example Code

Refer to the Python scripts provided in the project for:

Data Preprocessing

PCA Dimensionality Reduction

HDBSCAN Clustering

Visualization and Evaluation

Expected Outcomes

Clustering of facial images into meaningful groups.

Identification of noise points in the dataset.

Representative images for each cluster.

Generalization analysis for new data.

Challenges and Solutions

Noisy Data:

Simulated by adding Gaussian noise to test robustness.

Results analyzed through changes in cluster structure and silhouette scores.

Distance Metrics:

Experimented with metrics like euclidean, manhattan, and cosine to optimize clustering.