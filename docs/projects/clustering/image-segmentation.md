# 图像颜色分割

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：高级</li>
      <li><strong>类型</strong>：聚类应用</li>
      <!-- <li><strong>预计时间</strong>：4-6小时</li> -->
      <li><strong>技能点</strong>：图像处理、K-Means聚类、特征提取、可视化</li>
      <li><strong>对应知识模块</strong>：<a href="/core/clustering/kmeans.html">聚类算法</a></li>
    </ul>
  </div>
</div>

## 项目背景

图像分割是计算机视觉中的一个基本任务，目的是将图像分割成多个区域或对象，以便于进一步分析和理解。颜色分割是图像分割的一种简单而有效的方法，它基于像素的颜色特征将图像分割成不同的区域。

聚类算法，特别是K-Means，是实现颜色分割的常用方法。通过将图像中的像素视为多维空间中的点（例如RGB或HSV颜色空间），K-Means可以将这些点分组成不同的簇，每个簇代表一种主要颜色。

在这个项目中，我们将使用K-Means算法对图像进行颜色分割，学习如何处理图像数据并应用聚类技术。

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>颜色分割不仅用于图像处理和计算机视觉，还广泛应用于图像压缩、内容识别和艺术风格转换。例如，GIF图像格式使用颜色量化（一种颜色分割技术）来减少颜色数量，从而减小文件大小。</p>
  </div>
</div>

## 项目目标

1. 学习如何处理和表示图像数据
2. 应用K-Means算法进行颜色分割
3. 探索不同颜色空间和参数对分割结果的影响
4. 实现图像颜色量化和主色调提取
5. 可视化和评估分割结果

## 实施步骤

### 步骤1：图像加载与预处理

首先，我们加载图像并进行必要的预处理。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
from PIL import Image
import os

# 加载图像
image_path = 'sample_image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间从BGR到RGB

# 显示原始图像
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')
plt.show()

# 获取图像尺寸
height, width, channels = image.shape
print(f"图像尺寸: {width}x{height}, 通道数: {channels}")

# 将图像重塑为二维数组，每行代表一个像素
pixels = image.reshape(-1, channels)
print(f"像素数据形状: {pixels.shape}")

# 查看前几个像素的RGB值
print("前5个像素的RGB值:")
print(pixels[:5])
```

### 步骤2：确定最佳簇数

使用肘部法则和轮廓系数确定最佳簇数（颜色数量）。

```python
# 为了提高计算效率，可以对大图像进行下采样
def downsample_image(image, factor=0.5):
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height))

# 下采样图像用于确定最佳簇数
if pixels.shape[0] > 100000:  # 如果像素数量太多
    downsampled_image = downsample_image(image, factor=0.3)
    downsampled_pixels = downsampled_image.reshape(-1, channels)
    print(f"下采样后的像素数据形状: {downsampled_pixels.shape}")
    sample_pixels = downsampled_pixels
else:
    sample_pixels = pixels

# 肘部法则
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(sample_pixels)
    inertia.append(kmeans.inertia_)
    
    # 计算轮廓系数 (对于大图像，可能需要进一步采样)
    if sample_pixels.shape[0] > 10000:
        sample_idx = np.random.choice(sample_pixels.shape[0], 10000, replace=False)
        sample_for_silhouette = sample_pixels[sample_idx]
        labels = kmeans.predict(sample_for_silhouette)
        silhouette_scores.append(silhouette_score(sample_for_silhouette, labels))
    else:
        silhouette_scores.append(silhouette_score(sample_pixels, kmeans.labels_))

# 可视化肘部法则和轮廓系数
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('颜色数量 (k)')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('颜色数量 (k)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法')
plt.grid(True)
plt.tight_layout()
plt.show()

# 根据结果选择最佳簇数
best_k = k_range[np.argmax(silhouette_scores)]
print(f"最佳颜色数量 (基于轮廓系数): {best_k}")
```

### 步骤3：应用K-Means进行颜色分割

使用确定的最佳簇数应用K-Means进行颜色分割。

```python
# 应用K-Means聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(pixels)

# 获取聚类中心（代表主要颜色）
colors = kmeans.cluster_centers_.astype(int)
print("主要颜色 (RGB):")
for i, color in enumerate(colors):
    print(f"颜色 {i+1}: {color}")

# 为每个像素分配最近的聚类中心
labels = kmeans.predict(pixels)

# 创建分割后的图像
segmented_image = colors[labels].reshape(image.shape)
segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

# 显示分割结果
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f'颜色分割 (k={best_k})')
plt.axis('off')
plt.tight_layout()
plt.show()

# 显示主要颜色
plt.figure(figsize=(12, 2))
for i, color in enumerate(colors):
    plt.subplot(1, len(colors), i+1)
    plt.axis('off')
    plt.imshow([[color]])
    plt.title(f'颜色 {i+1}')
plt.tight_layout()
plt.show()
```

### 步骤4：颜色量化与主色调提取

实现图像颜色量化和主色调提取。

```python
# 计算每种颜色的像素比例
color_counts = np.bincount(labels)
color_percentages = color_counts / len(labels) * 100

# 按比例排序颜色
sorted_indices = np.argsort(color_percentages)[::-1]
sorted_colors = colors[sorted_indices]
sorted_percentages = color_percentages[sorted_indices]

# 显示颜色比例
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_percentages)), sorted_percentages, color=[np.clip(c/255, 0, 1) for c in sorted_colors])
plt.xlabel('颜色索引')
plt.ylabel('像素百分比 (%)')
plt.title('各主要颜色的像素比例')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# 创建颜色调色板
plt.figure(figsize=(12, 4))
for i, (color, percentage) in enumerate(zip(sorted_colors, sorted_percentages)):
    plt.subplot(1, len(sorted_colors), i+1)
    plt.axis('off')
    plt.imshow([[color]])
    plt.title(f'{percentage:.1f}%')
plt.tight_layout()
plt.show()

# 创建主色调提取结果
palette_height = 100
palette_width = width
palette = np.zeros((palette_height, palette_width, channels), dtype=np.uint8)

# 根据颜色比例分配宽度
start_x = 0
for color, percentage in zip(sorted_colors, sorted_percentages):
    end_x = start_x + int(palette_width * percentage / 100)
    palette[:, start_x:end_x] = color
    start_x = end_x

# 显示原始图像和提取的主色调
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(palette)
plt.title('提取的主色调')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### 步骤5：探索不同颜色空间的影响

比较RGB和HSV颜色空间对分割结果的影响。

```python
# 转换图像到HSV颜色空间
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
pixels_hsv = image_hsv.reshape(-1, channels)

# 应用K-Means聚类到HSV空间
kmeans_hsv = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_hsv.fit(pixels_hsv)

# 获取HSV空间的聚类中心
colors_hsv = kmeans_hsv.cluster_centers_.astype(int)

# 为每个像素分配最近的聚类中心
labels_hsv = kmeans_hsv.predict(pixels_hsv)

# 创建HSV分割后的图像
segmented_image_hsv = colors_hsv[labels_hsv].reshape(image_hsv.shape)
segmented_image_hsv = np.clip(segmented_image_hsv, 0, 255).astype(np.uint8)

# 转换回RGB空间进行显示
segmented_image_hsv_rgb = cv2.cvtColor(segmented_image_hsv, cv2.COLOR_HSV2RGB)

# 显示RGB和HSV分割结果比较
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_image)
plt.title(f'RGB颜色分割 (k={best_k})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV颜色空间')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(segmented_image_hsv_rgb)
plt.title(f'HSV颜色分割 (k={best_k})')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 步骤6：保存结果

保存分割结果和颜色调色板。

```python
# 保存分割后的图像
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 保存RGB分割结果
cv2.imwrite(os.path.join(output_dir, 'segmented_rgb.jpg'), 
            cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# 保存HSV分割结果
cv2.imwrite(os.path.join(output_dir, 'segmented_hsv.jpg'), 
            cv2.cvtColor(segmented_image_hsv_rgb, cv2.COLOR_RGB2BGR))

# 保存颜色调色板
cv2.imwrite(os.path.join(output_dir, 'color_palette.jpg'), 
            cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))

print(f"结果已保存到 {output_dir} 目录")
```

## 结果分析

通过K-Means聚类算法，我们成功地将图像分割成几个主要颜色区域。这种颜色分割方法可以有效地提取图像的主要颜色特征，减少颜色数量，同时保留图像的视觉结构。

比较RGB和HSV颜色空间的结果，我们发现HSV空间通常能更好地捕捉人类感知的颜色差异，因为它将色调、饱和度和亮度分开表示。这在某些应用中可能更有用，例如在不同光照条件下的颜色识别。

颜色量化结果显示了图像中各主要颜色的比例，这可以用于图像内容分析、风格识别和图像检索等应用。

## 进阶挑战

如果你已经完成了基本任务，可以尝试以下进阶挑战：

1. **空间感知分割**：结合像素位置信息进行分割，使用SLIC或Watershed等算法
2. **自适应簇数选择**：实现自动确定最佳颜色数量的方法
3. **图像风格转换**：基于颜色分割结果实现卡通化或油画风格转换
4. **颜色协调分析**：分析提取的颜色是否符合色彩理论中的协调原则
5. **批量处理**：开发一个能够批量处理多张图像的系统，并比较它们的颜色特征

## 小结与反思

通过这个项目，我们学习了如何使用K-Means聚类算法对图像进行颜色分割，并提取主要颜色特征。这种技术在图像处理、计算机视觉和设计领域有广泛的应用。

颜色分割是一种简单而强大的图像分析方法，它可以帮助我们理解图像的颜色组成，提取视觉特征，并为更复杂的图像处理任务提供基础。通过调整参数和选择不同的颜色空间，我们可以获得不同的分割效果，适应不同的应用需求。

### 思考问题

1. 颜色分割在哪些实际应用中最有价值？例如，在设计、医学图像分析或内容识别中如何应用？
2. K-Means算法的局限性是什么？在什么情况下它可能无法提供良好的颜色分割结果？
3. 如何将颜色分割与其他图像特征（如纹理、形状）结合，以获得更全面的图像理解？

<div class="practice-link">
  <a href="/projects/regression/house-price.html" class="button">下一个模块：预测与回归项目</a>
</div> 