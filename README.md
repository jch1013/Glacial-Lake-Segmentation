# 🏔️ Glacial Lake Segmentor
Using AI to identify glacial lakes from satellite imagery.

## Overview
**Glacial Lake Segmentor** is a machine learning project designed to detect glacial lakes from satellite images using semantic segmentation techniques. Born from my love for hiking and alpine landscapes, this project is both a personal exploration tool and a technical deep-dive into geospatial AI.

The goal was to automate the discovery of new glacial lakes — potential hiking spots — while building practical experience with deep learning, remote sensing data, and image segmentation workflows.

## Key Features
- 🛰️ Integration with [Sentinel Hub](https://www.sentinel-hub.com/) API to fetch high-resolution satellite imagery
- 🧠 UNet-based image segmentation using PyTorch and TensorFlow/Keras
- 🧹 Preprocessing pipeline for satellite images (e.g., cloud filtering, normalization)
- 🗺️ Output masks highlighting likely glacial lake regions
- 🧪 Model evaluation on hold-out datasets with basic metrics (IoU, Dice score)

## Tech Stack
- **Languages & Frameworks:** Python, TensorFlow, Keras, PyTorch
- **ML & Data:** NumPy, SciPy, scikit-learn, OpenCV
- **APIs & Tools:** Sentinel Hub API, matplotlib

## Screenshots
> _Coming soon_: visual examples of input satellite images alongside predicted glacial lake masks.

## What I Learned
- Developed a deeper understanding of semantic segmentation architectures, especially UNet
- Gained practical experience working with real-world satellite imagery and remote sensing data
- Learned to integrate disparate tools like satellite APIs, geospatial libraries, and deep learning frameworks into a cohesive workflow

## Future Improvements
- 🔁 Train on a larger and more geographically diverse dataset
- 🧠 Further model fine-tuning and hyperparameter optimization
- 🌦️ Add cloud masking and weather condition filtering
- 🌍 Potential integration with GIS platforms for route planning

## Why I Built This
As someone who enjoys exploring alpine regions, I wanted a tool that could help me discover new glacial lakes and backcountry destinations. This project let me combine that passion with my growing interest in machine learning and geospatial analysis.

