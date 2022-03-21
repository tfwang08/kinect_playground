# Introduction

This is my kinect playground project.

# Requirements

Step1. Install YOLOX from source.

```bash
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

Step2. Install Kinetic SDK.

```bash
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt-get update
```

```bash
sudo apt install k4a-tools
```

# Detection with YOLOX

```bash
python azure_kinect_realtime_detection.py
```

# Detection and measure how far the object is from the camera.

```bash
pythonn azure_kinect_how_far.py
```