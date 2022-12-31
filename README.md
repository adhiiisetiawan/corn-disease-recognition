<div align="center">

# Corn Disease Recognition

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-FF0000?logo=keras&logoColor=white)](https://keras.io/)
[![Android](https://img.shields.io/badge/Android-3DDC84?logo=android&logoColor=white)](https://www.android.com)
  
</div>

## Description

Corn is a major staple crop that is grown around the world. However, it is vulnerable to various diseases that can significantly impact the yield and quality of the crop. Early detection and treatment of these diseases is crucial for maintaining the health of the crop and ensuring a good harvest.

This project aims to develop a machine learning model for recognizing diseases in corn leaves. The model was trained on a dataset of images of infected and healthy corn leaves, collected and labeled by Acharya (October 2020)

The ultimate goal of this project is to provide a tool for farmers and agricultural professionals to quickly and accurately identify diseases in their corn crops, allowing them to take timely action to prevent the spread of the disease and protect their crops.
<br>
<br>
The project will include the following steps:

1. **Data collection:** The dataset for this project was downloaded from [Kaggle](https://www.kaggle.com/datasets/qramkrishna/corn-leaf-infection-dataset) (Acharya, October 2020).
2. **Data preprocessing:** Clean and prepare the data for modeling, including resizing and normalizing the images.
3. **Model training:** Train a machine learning model on the prepared dataset using supervised learning techniques.
4. **Model evaluation:** Evaluate the performance of the trained model on a separate test dataset to assess its accuracy and reliability.
5. **Android app development:** Develop an Android app that allows users to capture and classify images of corn leaves using the trained model.

This project is suitable for individuals with an interest in agriculture and machine learning, and those looking to develop skills in image classification, Android app development, and disease detection

## Prerequisites
Before training the model and deploying the app, you will need to install the following dependencies:

1. Python 3.6 or higher
2. TensorFlow 2.0 or higher
3. Android Studio 4.0 or higher [OPTIONAL]


## How to run

If you want to retrain, you can following this step.

Install dependencies

```bash
# clone project
git clone https://github.com/adhiiisetiawan/corn-disease-recognition
cd corn-disease-recognition

# [OPTIONAL] create python environment
python3 -m venv [your-python-venv-name]

# install requirements
pip3 install -r requirements.txt
```

Train model with default configuration

```bash
python3 main.py -c configs/corn_disease_config.json
```

Train model with chosen experiment configuration, you can copy default configuration and change any stuff there such as, learning rate, optimizer, batch size, etc. You also can change with your custom dataset by change in `configuration.json` file. Just change in `train_dir` and `val_dir` key. And you can train with your personal configuration.

```bash
python3 main.py -c configs/[your-personal-config].json
```

## Android App Development
An Android app was developed to allow users to capture and classify images of corn leaves using the trained model. The app includes a user-friendly interface for capturing and uploading images, as well as displaying the results of the classification. You can download the android apps [here](https://github.com/adhiiisetiawan/corn-disease-recognition/blob/main/CornDiseaseRecognitionApp/app/release/Corn%20Disease%20Recognition%20App.apk). 

### Export APK Manually [OPTIONAL]
You also can export APK manually by opening [CornDiseaseRecognitionApp](https://github.com/adhiiisetiawan/corn-disease-recognition/tree/main/CornDiseaseRecognitionApp) in Android Studio and follow this step. 
1. Open the project in Android Studio.
2. Go to "Build" > "Generate Signed APK" in the menu.
3. Follow the prompts to create a signed APK


## Reference
* Dataset: [Corn Leaf Infection Dataset Version 1](https://www.kaggle.com/datasets/qramkrishna/corn-leaf-infection-dataset) by Acharya (October 2020).
* Project Template: [Keras Project Template](https://github.com/Ahmkel/Keras-Project-Template) developed by [Ahmkel](https://github.com/Ahmkel).
