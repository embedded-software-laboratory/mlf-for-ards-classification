# MLP Framwork

This repository contains a software which can be used to
* train new machine learning models for the automated detection of ARDS,
* run and compare the previously trained models.

It aims to simplify the developement process for new models by providing a framework which contains contains the necessary steps that must be followed in order to 
train a new model.

The software can be divided into two parts: The part for timeseries models and the part for image models. Timeseries models are models which are trained with
regulary measured data like vitalparameters (e.g. heart rate, respiratory rate etc.) or laboratory values. Image models are trained to detect ARDS in X-Ray images
of the patient lung. 

A general machine learning process can be roughly divided into the following five steps:
* Data Extraction
* Data Preprocessing
* Data Transformation
* Model Training
* Model Evaluation.

The framework provides functionalities for the last 4 steps. For the first step, use the data-extractor (https://git-ce.rwth-aachen.de/smith-project/ARDS-MLP/data-basis/data-extraction)
for extracting timeseries data. The image data can be found in the SMITH Coscine Project. 

To prevent compatibility issues and minimalize the installation expense, Anaconda was used to provide an environment which contains all necessary python packages.
A manual for the installation of the framework can be found under "doc/Programm ausführen.md".

In the folder "src", you will find a file named "config.yml", which is used to set which steps the framework should execute and how they should be executed. A manual
on how to use this config file can be found under "doc/Anleitung Config-Datei.md".

In the folder "Save", you will find any outputs of the framework (if you do not change the output paths in the config file). 

If you want to add a new timeseries model into the framework, please refer to the manual at "doc/Neues Zeitreihenmodell hinzufügen.md". Unfortunately, there does
not exist a manual to add new image models to this point. 
