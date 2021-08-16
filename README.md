# Face Recognition system

## Table of contents
* [General info](#general-info)
* [Prerequisite](#prerequisite)
* [Setup](#setup)
* [How to use](#howtouse)

## General info
This project is a simple face detection system that uses 468 face mesh to find the identity of any face. 
To customize the data, basically you just have to reassign the images in 'data' directory. 
The labels are based on the directories inside. 
Make sure to use the same format (.jpg) for any images inside the data directory.
This example uses the faces of Tony Stark and Natasha Romanoff. 
The directory structure should be like this:
```
face_recognition
+---identities.pkl
+---README.md
+---run.py
+---data
¦   +---Natasha Romanoff
¦   +---Tony Stark
+---src
¦   +---utils.py
+---static
    +---1.png
    +---2.png
```
It originally uses Mediapipe Face Mesh algorithm by Google.<br>
https://mediapipe.dev/ <br>

## Prerequisite
Before using the program, make sure you already have 
installed the following software:
* Anaconda Navigator

## Setup
To run this project, open the Anaconda Prompt then run the following commands:
```
$ cd <project folder>
$ conda env create -f face_recognition.yml
$ conda activate face_recognition
$ python run.py --source 0
```
Make sure to select the right index of imaging device (by default is 0). If you use any webcam then the indext might be 1 or etc.

## How to use
If all the steps are done correctly, the program should appear like this.
![first look](static/first_look.png)
The "Run" option will directly infer the stored model, meanwhile the "Retrain Model" option will generate the new model that stores the provided dataset inside "data" folder. 
If the new model generated succesfully, the files inside 'data' directory are no longer needed since it automatically store the keypoints needed. 
For the first use, select Retrain Model, then Run. 
To reset all the secret inventory and personal stuffs count, select the Reset Logs button.
