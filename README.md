<div align="center">

# Corn Disease Recognition [Under Construction of Documentation]

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

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Reference
* Dataset: [Corn Leaf Infection Dataset Version 1](https://www.kaggle.com/datasets/qramkrishna/corn-leaf-infection-dataset) by Acharya (October 2020).
* Project Template: [Keras Project Template](https://github.com/Ahmkel/Keras-Project-Template) developed by [Ahmkel](https://github.com/Ahmkel).
