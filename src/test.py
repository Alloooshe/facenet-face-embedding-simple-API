from recognizer_facenet import FaceRecognizerFaceNet
import cv2
import numpy as np 

embedder = FaceRecognizerFaceNet()
embedder.load()
image = cv2.imread('Path/to/face/image')
image = np.expand_dims(image,axis=0)
res =embedder.extract_features(image)
embedder.clean()
print(res.shape)