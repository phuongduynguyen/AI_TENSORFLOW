import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

img = load_img(r'D:\DOANTOTNGHIEP\AI_FaceReg_TFLITE\dataset_v1\angelina_jolie\498.jpg')
img = img_to_array(img)
data = expand_dims(img, 0)
path="Newfolder"
count=0
# Dinh nghia 1 doi tuong Data Generator voi bien phap chinh sua anh lat ngang doc
myImageGen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.1,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
# Batch_Size= 1 -> Moi lan sinh ra 1 anh
gen = myImageGen.flow(data, batch_size=32)
# Sinh ra 9 anh va hien thi len man hinh
for i in range(9):
        count = count + 1
        myBatch = gen.next()
        image = myBatch[0].astype('uint8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path+"/"+str(count)+".jpg",gray)


print(count)
