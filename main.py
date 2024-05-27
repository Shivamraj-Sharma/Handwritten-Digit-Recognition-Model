import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('model.keras')

img_number = 1
while os.path.isfile(f"Digits/digit{img_number}.png"):
    try:
        img = cv2.imread(f"Digits/digit{img_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably {np.argmax(prediction)}")
        plt.imshow(img[0],plt.cm.binary)
        plt.show()
    except Exception as e:
        print(e)
    finally:
        img_number += 1