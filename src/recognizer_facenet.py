from  recognizer_base import FaceRecognizer
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import re
from tensorflow.python.platform import gfile
import os
import numpy as np 


class FaceRecognizerFaceNet(FaceRecognizer):
     '''
     FaceRecognizerFaceNet impelements FaceReconizer and perform the embedding process using facenet model
     
     Attributes:
         isloaded (bool) : True if the model is loaded and false otherwise
         sess (tensorflow session) : the local session to use when performing operations
         model_dir (string) : diractory to facenet model 
     '''
     NAME = 'recognizer facenet'
     
     def __init__(self):
         super(FaceRecognizer, self).__init__()
         self.is_loaded = False;
         self.sess =None
         self.model_dir=os.path.join(  os.path.dirname(os.path.realpath(__file__)),"..","model","facenetmodel.pb") 
         
         
     def __str__(self):
        return self.name()

     def name(self):
        return 'facenet recognizer'

     def load(self):
       '''
        load the wights of facenet model to prepare for use it also initialize the sess attribute 
        
        Returns:
            True if the model was loaded successfully 
       '''
       if  self.is_loaded:
            print('model is already loaded')
            return True
         
       gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
       self.sess= tf.Session(config=tf.ConfigProto(device_count={ "CPU":1,"GPU":1},
                                               gpu_options=gpu_options
                                               ))   
       default_graph = self.load_graph(self.model_dir)
        
       self.f_input = default_graph.get_tensor_by_name('input:0')
       self.f_input_2 =  default_graph.get_tensor_by_name('phase_train:0')
       self.f_output = default_graph.get_tensor_by_name('embeddings:0')
              

       self.sess = tf.Session(graph=default_graph)            
       self.is_loaded= True
       print('facenet model was loaded successfully.... ')
       return True
    
    
    
    
     def load_graph(self,model_path): 
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

     def configure(self,model_dir):
       '''
       
       set the configuration of facenet model 
       
       Args:
         minsize (int) : the minimum size of detected faces
         threshold (list) : the float values to use in filtering the faces smaller values means smaller confidance
         facetor (float) : the scaling factor of the face 
         
      
       '''
       self.model_dir=model_dir



     def extract_features(self, image,clean=False):
        '''
        
        embeds face image into vector of 512 value that represent the face using the facenet model 
        
        Args :
            image (list numpy array) : the input aligned face image or images of shape (n,3,160,160)
            clean (boolean) : if true it clean the loaded model 
            
        Returns : 
            parsed feature vector of shape (n,512)
        
        '''
        
        if not self.is_loaded :
            self.load()
        image = self.prewhiten(image)

        with self.sess.as_default():


            # Run forward pass to calculate embeddings
            feed_dict = { self.f_input: image, self.f_input_2:False }
            emb =  self.sess.run(self.f_output, feed_dict=feed_dict)
     
        if clean : 
            self.clean()
        return self.parse_output(emb)
    
    
    
    
     def parse_output(self,emb):
        '''
         helper function to parse ouput of the recognizer to fit in the pipline - can be ignored 
        '''
        return emb
        
     def clean(self):
        '''
         clean the loaded model 
         
        '''
        self.sess=None
        return True

     def prewhiten(self, x):
        '''
        
        helper function to normalize the input image by sutracting the mean and divide on the std so all values are between 0 and 1 
        
        Args :
            x (numpy array) : input array 
        
        Returns: 
            y (numpy array) : normalized input 
        
        '''
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  