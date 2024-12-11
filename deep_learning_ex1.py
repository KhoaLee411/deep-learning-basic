import numpy as np
from matplotlib import pyplot as plt
from seaborn.external.husl import dot_product

file_path = "./full_numpy_bitmap_bed.npy"  # <= Các bạn thay từ bicycle bằng tên tương ứng của category các bạn chọn nhé
images = np.load(file_path).astype(np.float32)  # Load toàn bộ các ảnh của category này vào biến images
# print(images.shape)
# train_images = images[:-10]  # Lấy tất cả ảnh, ngoại trừ 10 ảnh cuối ra làm bộ training.
test_images = images[-10:]  # Giữ 10 ảnh cuối làm bộ test
#
# avg_image = np.mean(train_images, axis=0)

# print(avg_images.shape) =>  kết quả cho thấy avg_images đang là mảng một chiều
# cách biến mảng một chiều thành mảng hai chiều biến (784,) thành mảng hai chiều 28*28

# avg_image = np.reshape(avg_image, (28,28))
# print(avg_image.shape)

# plt.imshow(avg_image)
# plt.show()

index = 1
test_image = test_images[index]

# # tính tích vô hướng của hai vecto 1 chiều
# score = np.dot(test_image, avg_image)
# print(score)
# print(test_image @ avg_image)

# tích vô hướng của hai vecto thể hiện điều gì ? ====> Tích vô hướng là một con số, thể hiện sự tương quan về mặt hướng và độ lớn của hai vecto đó

categories = ["banana","apple","book","bed"]
scores = []
avg_images = []
for c in categories:
    file_path = f"./full_numpy_bitmap_{c}.npy"
    images = np.load(file_path).astype(np.float32)
    avg_image = np.mean(images, axis=0)
    avg_images.append(avg_image.reshape(28,28))
    dot_prod = test_image @ avg_image
    scores.append(dot_prod)

plt.figure(figsize=(10,4))
for i in range(len(categories)):
    plt.subplot(2, 5, i+1)
    plt.imshow(avg_images[i])
    plt.title(categories[i])
plt.show()
# End