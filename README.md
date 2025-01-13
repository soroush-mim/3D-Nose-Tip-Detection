# 3D Nose Tip Detection in Facial Point Clouds

## Project Overview

This project implements two methods for detecting the nose tip in 3D facial point clouds using the [d3dfacs_alignments](https://files.is.tue.mpg.de/tbolkart/FLAME/d3dfacs_alignments.zip) dataset:

1. **Protrusion-Based Detection**: Identifies the most protruding point in the facial point cloud, assuming it corresponds to the nose tip.

2. **DBSCAN Clustering-Based Detection**: Utilizes the DBSCAN clustering algorithm to segment facial regions and identify the nose region based on spatial characteristics. Then, it will calculate the most protruding point in the nose cluster. also calculates the X-axis based on the two right and left points of the nose. 

For the data, I sampled some files from the d3dfacs_alignments dataset and labeled the nose tip manually. You can see this data in the Data folder.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/soroush-mim/3D-Nose-Tip-Detection.git
   cd 3D-Nose-Tip-Detection
2. **Set Up a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
4. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt

## Usage

1. **for a single ply file**:

   ```bash
   python main.py --path sample.ply
   #it will calculate the nose tip using 2 solutions and plot these 2 predicted points and the real nose tip
   #on the 2D picture of the face. It also calculates the inference time for each solution.
   
2. **for a directory:**

   ```bash
   python main.py --path Data
   #for all ply files in the Data folder, it  will calculate the nose tip using 2 solutions.
   #Then, calculate each solution's MAE and average inference time.
