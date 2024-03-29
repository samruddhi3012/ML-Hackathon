# ML-Hackathon
<h2>Objective :</h2>
<p>To develop a gender and age detection system that can accurately determine the gender and age of the person (face) from human face images and real-time video streams from mobile camera.</p>

<h2>About the Project :</h2>
<p>We build deep learning models to predict age and gender from face pictures and real time video streams, which make use of TensorFlow and Keras. In this project, we have used MobileNetV2 as a foundational model, augmenting it with more thick layers. Custom data generators are used to apply data augmentation strategies during training. Appropriate optimization strategies, loss functions, and assessment measures are integrated into the models. To make training and monitoring more efficient, we've included callbacks like ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau. 
The models have been trained on kaggle using Tesla P100 GPU and inferencing is performed on google colab.

### Specification of Gender Model 
- Accuracy: 93.41%  
- Loss: 0.1737 
- Time taken to train: 44.8 mins

### Specification of Age Model 
- Loss: 48.1983
- Mean absolute error: 4.9132
- Time taken to train: 41.2 mins
- 
### Machine specification: 
- Google Colab (Intel Xeon CPU with 2vCPUs and 13GB RAM)
- Tesla P100 16GB GPU
- Kaggle
</p>

<h2>Dataset :</h2>
<p>For this project, we have used the UTKFace dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/datasets/jangedoo/utkface-new">here</a>. 

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. The models I used had been trained on this dataset.</p>

<h2>Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>TensorFlow</li>
  
       pip install tensorflow
</ul>
<ul>
 <li>Keras</li>
  
       pip install keras
</ul>

<h2>The contents of this Project :</h2>
<ul>
  <li>GenderAgeInference.ipynb</li>
    <ul> The code implements gender and age prediction from facial images using TensorFlow and Keras. It defines models for gender and age prediction using the MobileNetV2 architecture, loads pre-trained weights, and provides utility functions for image processing, face detection, and prediction visualization. Sample images are processed, faces are detected, and gender and age predictions are made, with bounding boxes and labels drawn on the images. Overall, it showcases deep learning techniques for facial analysis in a concise and efficient manner.
    </ul>
  <li>A few pictures for training and testing of model.</li>
  <li>topage_weights.keras</li>
    <ul> https://drive.google.com/file/d/1LoWfP2epBJUcdLkaYamJSg7kHHZhhMAH/view?usp=drivesdk </ul>
  <li>topgender_weights.keras</li>
    <ul>https://drive.google.com/file/d/1LVEYwvXk3-qeNgKF_x1s-ccLTjv3Jg2o/view?usp=drivesdk
    </ul>

  <li>gender-and-age-prediction.ipynb</li>
    <ul> The Jupyter notebook presents a comprehensive deep learning pipeline designed for gender and age inference from images employing TensorFlow/Keras. The process encompasses image data preprocessing, segmentation into training, validation, and testing subsets, and the establishment of data generators to facilitate batch processing. Two distinct models are constructed for gender and age prediction, leveraging pre-trained MobileNetV2 architectures as feature extractors. Model training incorporates callback mechanisms for model checkpointing and early stopping. Upon evaluation with test data, the models demonstrate promising performance, achieving high accuracy of 93.41% in gender prediction and minimal mean absolute error of 4.9132 in age estimation.
    </ul>
 </ul>
 
