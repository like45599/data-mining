# Image Color Segmentation

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Advanced</li>
      <li><strong>Type</strong>: Clustering Application</li>
      <!-- <li><strong>Estimated Time</strong>: 4-6 hours</li> -->
      <li><strong>Skills</strong>: Image Processing, K-Means Clustering, Feature Extraction, Visualization</li>
      <li><strong>Relevant Knowledge Module</strong>: <a href="/en/core/clustering/kmeans.html">Clustering Algorithms</a></li>
    </ul>
  </div>
</div>

## Project Background

Image segmentation is a fundamental task in computer vision, aimed at dividing an image into multiple regions or objects for further analysis and understanding. Color segmentation is a simple and effective method of image segmentation, which divides an image into different regions based on pixel color features.

Clustering algorithms, particularly K-Means, are commonly used for color segmentation. By treating the pixels of an image as points in a multidimensional space (such as RGB or HSV color space), K-Means can group these points into different clusters, each representing a dominant color.

In this project, we will use the K-Means algorithm for color segmentation of an image, learning how to process image data and apply clustering techniques.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Color segmentation is not only used in image processing and computer vision but is also widely applied in image compression, content recognition, and artistic style transfer. For example, the GIF image format uses color quantization (a type of color segmentation technique) to reduce the number of colors, thus reducing the file size.</p>
  </div>
</div>

## Project Goals

1. Learn how to process and represent image data
2. Apply the K-Means algorithm for color segmentation
3. Explore the impact of different color spaces and parameters on the segmentation results
4. Implement image color quantization and dominant color extraction
5. Visualize and evaluate the segmentation results

## Implementation Steps

### Step 1: Image Loading and Preprocessing

First, we load the image and perform necessary preprocessing.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
from PIL import Image
import os

# Load image
image_path = 'sample_image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color space from BGR to RGB

# Display original image
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Get image size
height, width, channels = image.shape
print(f"Image size: {width}x{height}, Channels: {channels}")

# Reshape image into a 2D array, each row represents a pixel
pixels = image.reshape(-1, channels)
print(f"Pixel data shape: {pixels.shape}")

# View RGB values of the first few pixels
print("First 5 pixels RGB values:")
print(pixels[:5])
```

### Step 2: Determining the Best Number of Clusters

Use the Elbow Method and Silhouette Coefficient to determine the optimal number of clusters (colors).

```python
# To improve computational efficiency, downsample large images
def downsample_image(image, factor=0.5):
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height))

# Downsample image for determining the best number of clusters
if pixels.shape[0] > 100000:  # If too many pixels
    downsampled_image = downsample_image(image, factor=0.3)
    downsampled_pixels = downsampled_image.reshape(-1, channels)
    print(f"Downsampled pixel data shape: {downsampled_pixels.shape}")
    sample_pixels = downsampled_pixels
else:
    sample_pixels = pixels

# Elbow Method
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(sample_pixels)
    inertia.append(kmeans.inertia_)
    
    # Calculate silhouette score (for large images, further sampling may be needed)
    if sample_pixels.shape[0] > 10000:
        sample_idx = np.random.choice(sample_pixels.shape[0], 10000, replace=False)
        sample_for_silhouette = sample_pixels[sample_idx]
        labels = kmeans.predict(sample_for_silhouette)
        silhouette_scores.append(silhouette_score(sample_for_silhouette, labels))
    else:
        silhouette_scores.append(silhouette_score(sample_pixels, kmeans.labels_))

# Visualize Elbow Method and Silhouette Scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Colors (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Number of Colors (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Coefficient Method')
plt.grid(True)
plt.tight_layout()
plt.show()

# Select the best number of clusters based on Silhouette Score
best_k = k_range[np.argmax(silhouette_scores)]
print(f"Best number of colors (based on silhouette score): {best_k}")
```

### Step 3: Apply K-Means for Color Segmentation

Use the determined optimal number of clusters to apply K-Means for color segmentation.

```python
# Apply K-Means clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(pixels)

# Get cluster centers (representing the main colors)
colors = kmeans.cluster_centers_.astype(int)
print("Main Colors (RGB):")
for i, color in enumerate(colors):
    print(f"Color {i+1}: {color}")

# Assign the closest cluster center to each pixel
labels = kmeans.predict(pixels)

# Create segmented image
segmented_image = colors[labels].reshape(image.shape)
segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

# Display segmented result
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f'Color Segmentation (k={best_k})')
plt.axis('off')
plt.tight_layout()
plt.show()

# Display main colors
plt.figure(figsize=(12, 2))
for i, color in enumerate(colors):
    plt.subplot(1, len(colors), i+1)
    plt.axis('off')
    plt.imshow([[color]])
    plt.title(f'Color {i+1}')
plt.tight_layout()
plt.show()
```

### Step 4: Color Quantization and Dominant Color Extraction

Implement image color quantization and dominant color extraction.

```python
# Calculate pixel proportions for each color
color_counts = np.bincount(labels)
color_percentages = color_counts / len(labels) * 100

# Sort colors by proportion
sorted_indices = np.argsort(color_percentages)[::-1]
sorted_colors = colors[sorted_indices]
sorted_percentages = color_percentages[sorted_indices]

# Display color proportions
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_percentages)), sorted_percentages, color=[np.clip(c/255, 0, 1) for c in sorted_colors])
plt.xlabel('Color Index')
plt.ylabel('Pixel Percentage (%)')
plt.title('Pixel Proportion for Each Main Color')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Create color palette
plt.figure(figsize=(12, 4))
for i, (color, percentage) in enumerate(zip(sorted_colors, sorted_percentages)):
    plt.subplot(1, len(sorted_colors), i+1)
    plt.axis('off')
    plt.imshow([[color]])
    plt.title(f'{percentage:.1f}%')
plt.tight_layout()
plt.show()

# Create dominant color extraction result
palette_height = 100
palette_width = width
palette = np.zeros((palette_height, palette_width, channels), dtype=np.uint8)

# Assign width according to color proportion
start_x = 0
for color, percentage in zip(sorted_colors, sorted_percentages):
    end_x = start_x + int(palette_width * percentage / 100)
    palette[:, start_x:end_x] = color
    start_x = end_x

# Display original image and dominant color extraction
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(palette)
plt.title('Extracted Dominant Colors')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### Step 5: Explore the Impact of Different Color Spaces

Compare the segmentation results between RGB and HSV color spaces.

```python
# Convert image to HSV color space
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
pixels_hsv = image_hsv.reshape(-1, channels)

# Apply K-Means clustering to HSV space
kmeans_hsv = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_hsv.fit(pixels_hsv)

# Get cluster centers in HSV space
colors_hsv = kmeans_hsv.cluster_centers_.astype(int)

# Assign the closest cluster center to each pixel
labels_hsv = kmeans_hsv.predict(pixels_hsv)

# Create HSV segmented image
segmented_image_hsv = colors_hsv[labels_hsv].reshape(image_hsv.shape)
segmented_image_hsv = np.clip(segmented_image_hsv, 0, 255).astype(np.uint8)

# Convert back to RGB for display
segmented_image_hsv_rgb = cv2.cvtColor(segmented_image_hsv, cv2.COLOR_HSV2RGB)

# Display RGB and HSV segmentation results comparison
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_image)
plt.title(f'RGB Color Segmentation (k={best_k})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV Color Space')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(segmented_image_hsv_rgb)
plt.title(f'HSV Color Segmentation (k={best_k})')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### Step 6: Save Results

Save the segmentation results and color palette.

```python
# Save segmented images
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save RGB segmentation result
cv2.imwrite(os.path.join(output_dir, 'segmented_rgb.jpg'), 
            cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# Save HSV segmentation result
cv2.imwrite(os.path.join(output_dir, 'segmented_hsv.jpg'), 
            cv2.cvtColor(segmented_image_hsv_rgb, cv2.COLOR_RGB2BGR))

# Save color palette
cv2.imwrite(os.path.join(output_dir, 'color_palette.jpg'), 
            cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))

print(f"Results saved to {output_dir} directory")
```

## Results Analysis

By applying the K-Means clustering algorithm, we successfully segmented the image into several main color regions. This color segmentation method effectively reduces the number of colors in an image while retaining its visual structure.

When comparing the results between the RGB and HSV color spaces, we found that the HSV space usually captures color differences more naturally, as it separates hue, saturation, and value. This can be more useful in applications like color recognition under varying lighting conditions.

The color quantization results show the proportion of each major color in the image, which can be applied to image content analysis, style recognition, and image retrieval.

## Advanced Challenges

If you have completed the basic tasks, try the following advanced challenges:

1. **Spatial-Aware Segmentation**: Combine pixel position information for segmentation, using algorithms like SLIC or Watershed
2. **Adaptive Cluster Number Selection**: Implement an automatic method for determining the best number of colors
3. **Image Style Transfer**: Implement cartoonization or oil painting style transfer based on color segmentation results
4. **Color Harmony Analysis**: Analyze whether the extracted colors follow color harmony principles from color theory
5. **Batch Processing**: Develop a system that can batch process multiple images and compare their color features

## Summary and Reflection

Through this project, we learned how to use the K-Means clustering algorithm for color segmentation and extract dominant color features from images. This technique is widely applicable in image processing, computer vision, and design.

Color segmentation is a simple yet powerful method for image analysis, helping us understand the color composition of an image, extract visual features, and provide a foundation for more complex image processing tasks. By adjusting parameters and selecting different color spaces, we can obtain varying segmentation results to meet different application needs.

### Reflection Questions

1. In which practical applications is color segmentation most valuable? How can it be applied in design, medical image analysis, or content recognition?
2. What are the limitations of the K-Means algorithm? In what situations might it not provide good color segmentation results?
3. How can color segmentation be combined with other image features (such as texture, shape) for a more comprehensive image understanding?

<div class="practice-link">
  <a href="/en/projects/regression/house-price.html" class="button">Next Module: Prediction and Regression Project</a>
</div>
