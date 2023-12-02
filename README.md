Deep Learning Website Address:
https://helix-evening-373.notion.site/Deep-Learning-32bb4735ebc2470ba1cde379c7e77ce0

Deep Learning YouTube Link:
https://youtu.be/c6q8gl9f-N0

I. Introduction
Facial expressions are the most important method of expressing human emotional information (Cai and Wei, 2020; Zhang et al., 2021). Facial expression recognition has been recognized as an important part of the interaction between humans and computers because it is based on human emotions and thoughts. The specific process can be divided into three stages: face detection, feature extraction, and classification modules, among which face detection is the most important (Adjabi et al., 2020; Zhang et al., 2021). This topic has been utilized in various fields, including psychology. However, due to the complexity and variability of expressions, research related to expression emotion detection technology has included many limitations. But as big data has been secured and hardware has developed, various deep learning technologies that overcome the limitations of existing machine learning models have emerged. Therefore, this project aims to create a model that classifies human emotions with higher accuracy than past studies through convolutional neural networks. Detailed content about the algorithm is explained in the methodology section.

II. Datasets
The data consists of black and white facial images of 48x48 pixels. Faces are automatically registered so that the face is somewhat centered, and they occupy almost the same space in each image. The topic is to classify each face into one of seven categories (0=Anger, 1=Disgust, 2=Fear, 3=Happiness, 4=Sadness, 5=Surprise, 6=Neutral) based on the emotions expressed in facial expressions. The training set consists of 28,709 examples, and the public test set consists of 3,589 examples.
The CSV file consists of two main columns: "emotion" and "pixels". The "emotion" column contains a numeric code between 0 and 6 that represents the emotion in the image. The "pixels" column contains a string for each image. The content of this string is an arrangement of pixel values in the order of rows, separated by spaces. This dataset is composed of data collected as part of an ongoing research project by Pierre-Luck Carrier and Aaron Courville.
You can easily access and download the dataset by clicking on the website or file button below.
https://www.kaggle.com/datasets/ahmedmoorsy/facial-expression

III. Methodology
1. Data Normalization
In this project, we processed and normalized data consisting of 35,887 entries for more effective results. A key issue affecting accuracy was the lack of input data normalization. The pixel values of the images from the fer2013.csv dataset ranged between 0 and 255. To enhance accuracy, we normalized these values by dividing the entire pixel values by 255, adjusting the pixel values to have a minimum of 0 and a maximum of 1.
2. Data Reshape and Training & Validation Data Preparation
We reshaped the dataset (N, D = X.shape) into a format suitable for CNN processing and split it into training and validation sets using train_test_split(), ensuring a distribution ratio of 9:1.
3. Model Building and Training
CNN, a type of artificial neural network, is designed for pixel data processing and image recognition. It consists of layers like input, output, and hidden layers (comprising multiple convolutional layers, pooling layers, fully connected layers, and normalization layers), making it effective for image and natural language processing.
We built a CNN model using various layers like Conv2D, MaxPooling2D, Flatten, and Dense. The model was compiled and trained with training and validation data, saving the model after each epoch to 'model_filter.h5'.
4. Evaluation and Real-time Prediction
The model's performance was evaluated using the test dataset, and a real-time emotion detection code was developed using the trained model. The application utilized a laptop webcam to classify emotions based on facial expressions.

IV. Conclusion
This project utilized the fer2013.csv dataset, consisting of black and white facial images classified into seven categories based on facial expressions. We achieved more effective results through data preprocessing, normalization, and model training using CNN. However, the project faced limitations in data distribution and accuracy, providing valuable insights into machine learning and deep learning algorithms.
