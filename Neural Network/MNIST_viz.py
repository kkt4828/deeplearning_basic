from sklearn.datasets import fetch_openml
from PIL import Image # pillow library로 image로 변환할 때 사용하는 라이브러리
import numpy as np
mnist = fetch_openml('mnist_784')

img_array = np.array(mnist.data[:1])
print(img_array.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img_show(img_array.reshape(28, 28))


