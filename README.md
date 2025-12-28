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

![E12C434E-5A80-49E6-9B36-2AEA25C64A3C_1_102_o](https://github.com/user-attachments/assets/011f643e-f8f4-4155-8b52-ae0550c7a81d)




