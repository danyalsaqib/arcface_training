# arcface_training
A module for finetuning ArcFace, and logging hyperparameters and metrics via MLflow.

This repository contains files for finetuning of the retinaface pytorch module, alongside MLflow logging for the model, its parameters, and its metrics. Run the file 'train.py' using python to finetune the model, and log information: `python3 train.py`. The training dataset is under 'data/widerface/train'.
