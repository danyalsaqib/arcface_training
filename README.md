# arcface_training
A module for finetuning ArcFace, and logging hyperparameters and metrics via MLflow.

This repository contains files for finetuning of the arcface tensorflow module, alongside MLflow logging for the model, its parameters, and its metrics. Run the file 'keras_model_saving_script.py' using python to finetune the model, and log information: `python3 keras_model_saving_script.py`. The training dataset is under the 'celeb_images' folder.
