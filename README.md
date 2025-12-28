# DetectionRT

An ios application for real-time object detection using YOLO11n with MNN backend.

## Implementation
The project is build by XCode 26 and designed to support iOS 26. 

It includes 2 main functions: Camera detection (real-time) and photo detection.

We implement YOLO11n model with MNN backendn and support Metal acceleration.

The model will be loaded after installing and opening the application for the first time. A cache file for metal is then preserved to accelerate the initialization of next cold run.

## Installation
1. Clone this repository to local environment with XCode installed.

2. Change signature and developer of the project to support building.

3. Connect to an iOS devices and runnning. The simulator is not usable due to camera requirements.

## Contribute

This repository is largely dependent on MNN framework and YOLO11n. The main coding is inspired by ChatGPT5.2 and Google Gemini 3 pro. 

![CE015472-2E8A-4254-AA6A-5E60A9B3D4B3_4_5005_c](https://github.com/user-attachments/assets/34c19885-521d-4e9c-83aa-dea6bced28de)
![3711E40D-5B8B-4A80-B881-5721AC66CDDD_4_5005_c](https://github.com/user-attachments/assets/4dfd8731-b87a-4702-b41c-87734670fdab)
![EB2C6F3A-D80B-4F44-A837-80306FB097DD_4_5005_c](https://github.com/user-attachments/assets/c60c23bf-b11f-450f-873d-91b7aea34c53)





