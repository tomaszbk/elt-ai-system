#!/bin/bash

echo "Downloading dataset from Kaggle"

dataset_url="sagaraiarchitect/laptop-price-explorer-the-ml-model"

mkdir ~/.kaggle/ && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d ${dataset_url}

unzip laptop-price-explorer-the-ml-model.zip -d .