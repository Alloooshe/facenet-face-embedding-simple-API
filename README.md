# facenet-face-embedding-simple-API
a simple wrapper to use Facenet model to extract embeddings of a face image
this is a simple class wrapper for face embedding extraction using the famous **Facenet** model, the model used here is the frozen graph trained by @davidsandberg big thanks for his brilliant work.

the intent of this repository is to make face recognition very easy to use in projects. it works with TensorFlow 2 and TensorFlow 1.x can be used with GPU or CPU without modification. 


## How to use 

1- clone repo, copy files to your project. 

2- import wrapper class 

    from recognizer_facenet import  FaceRecognizerFaceNet

3- create an instance and load the model

    recognizer = FaceRecognizerFaceNet()
	 
4- extract 512 face features that can be used for face recognition, face identification ...etc

    embedding = recognizer.extract_features(image)
   
 note: you can extract features for a batch of images with one pass since the input shape is [n,?,?,3]
 
5- clean your model to save space 
 
    recognizer.clean()

you can check test.py file for complete example. 

## dependencies 

TensorFlow 2, or you can just use TensorFlow 1.x by changing import line in recognizer_facenet, because we use tensorflow.compat.v1
NumPy

have fun !!

