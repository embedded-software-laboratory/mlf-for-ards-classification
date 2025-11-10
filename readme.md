# ML Framwork

Test for git

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

## Getting Started
**Warning this setup only works if you have access to the git-ce repository of the ARDS-MLF for public release the container image has to be moved to a publically accesssible container registry and the .devcontainer/devcontainer.json image has to be adjusted!**

The first step in starting the development of the ARDS-MLF is to clone the repository to your local machine. This can be either done using ssh or using https. In order to utilize any of these options you first need to setup your git-ce account correctly.

### Cloning via ssh
If you want to clone the repository via ssh, you first need to create an ssh-key on your local machine. This can be done by typing "ssh-keygen" into the terminal of your choice and then following the process that is described in the terminal. After having created the key it can be found at "C:\Users\\\<username>\\.ssh" (Windows) or "~/.ssh/" (Linux, Mac). Open the file that ends with ".pub" and copy its content. Then open [this]("https://git-ce.rwth-aachen.de/-/user_settings/ssh_keys") page and click on "Add new key". Paste the content copied from the ".pub" file, choose an Expiration date and click "Add key". You should now be able to clone the repository by typing "git clone git@git-ce.rwth-aachen.de:smith-project/ARDS-MLP/ml-framework.git" into the terminal at your local machine and answering the following questions with yes.

### Clonig via https
In order to clone the repository via https. You first need to create an personal access token as described [here](#creating-access-tokens). Then clone the repository by typing "git clone https://git-ce.rwth-aachen.de/smith-project/ARDS-MLP/ml-framework.git" then use your git-ce username and the personal access token to complete the login.




### Getting Started Locally 

The recommended way to setup the development environment for this project is the devcontainer setup. Although conda environments were used for earlier version the environment.yml is not updated any more. 

#### Devcontainer
In order for the dev container to work you need to have docker installed. A guide on how to install docker can be found [here] ("https://docs.docker.com/desktop/").
After docker is installed you need to have a personal access token which you can use to download the needed docker image. The process is described [here](#creating-access-tokens). Watch out to set the correct scopes.
Then follow the next steps:
* Open the terminal on your local machine
* Type "docker login registry.git-ce.rwth-aachen.de" and then login with your git-ce username and access token.

For VS-Code:
* Install the devcontainer plugin [here]("vscode:extension/ms-vscode-remote.remote-containers").
* Open the folder containing the repository
* Press "Ctrl+Shift+P" and select "Dev Containers: Rebuild and Reopen in Container"
* Wait for the image to Download (This may take some time)
* Once VS Code show the workspace start Developing

For PyCharm:

* If you have an open project: Close the project
* Make sure you have the [Dev Containers Plugin]("https://plugins.jetbrains.com/plugin/21962-dev-containers") installed 
* On the welcome to PyCharm screen go to "Remote Development" and then "Dev Containers"
* Click on "New Dev Container
* Select "From Local Project"
* Under path to devcontainer.json navigate to the folder of this repository and then select the "devcontainer.json" file located in the ".devcontainer" folder. 
* Click "Build Container and Continue"
* Wait for the build process to end. Once PyCharm finishes building the container and opens the editor you can start developing.



### Getting Started CLAIX2023
In order to be able to utilize the RWTH HPC CLAIX2023 you need to have setup a access token with the correct scope to use devcontainers [guide](#creating-access-tokens).

Then follow the steps described below:
 * Login to a Claix login node
 * Execute "apptainer remote login --username <git-ce username> docker://registry.git-ce.rwth-aachen.de" enter your access token as the password.
 * Execute "apptainer pull \<image name\> docker://registry.git-ce.rwth-aachen.de/smith-project/ards-mlp/ml-framework:latest" This downloads the docker image to the hpc and creates an image usable for apptainer. Choose the image name as you like (it has to end with ".sif")
 * Start the container with "apptainer shell <image name>" if you need to use GPUS during the run of the framework append "--nv" after shell. Watchout to fully specify the path of the image if you are not in the same directory where the image is stored
 * The last command will open a shell in the container you may start the ARDS-MLF as you normally do via the command line. 

## Creating Access Tokens
Acess Tokens are created [here]("https://git-ce.rwth-aachen.de/-/user_settings/personal_access_tokens").
If you want to clone the repository via https make sure the token has the following scopes:
 * read_repository
 * write_repository

If you want to use the [devcontainer setup](#devcontainer) you need to add the following scope:
* read_registry

If you want to update the docker image for the devcontainer you also need the following scope:
* write_registry

Then click on "Create token", copy the resulting token and save it somewhere secure.

## Updating the docker image
If you need further dependencies in the docker image you need to update the docker file located in the root of the repository. For new python libraries follow the already existing examples. Please make sure you do not update any installed libraries unknowlingly as this may break dependencies.
 After having updated the Dockerfile build the new image by executing "docker build -t registry.git-ce.rwth-aachen.de/smith-project/ards-mlp/ml-framework:latest -t registry.git-ce.rwth-aachen.de/smith-project/ards-mlp/ml-framework:<nextversionnumber> ."
 After building the new image login to the git-ce container registry by executing "docker login registry.git-ce.rwth-aachen.de" using your git-ce username and access token. Then push the newly build image using "docker push registry.git-ce.rwth-aachen.de/smith-project/ards-mlp/ml-framework:latest" and "docker push registry.git-ce.rwth-aachen.de/smith-project/ards-mlp/ml-framework:<nextversionnumber>.


