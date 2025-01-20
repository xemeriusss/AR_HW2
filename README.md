YouTube demo: https://www.youtube.com/watch?v=2J75gCaqSI0

# Homework #2: Augmented Reality Point-Cloud Alignment

## Description
This project involves building a Unity application to align two point-cloud datasets. The program reads two files containing 3D point data, applies transformation algorithms, and visualizes the results.

## Features
1. **File Input**:
   - Reads two files containing 3D points in the format:
     ```
     num_pts
     x1 y1 z1
     ...
     xn yn zn
     ```
2. **Transformation Algorithms**:
   - Rigid transformation with RANSAC and three-point alignment methods.
   - Option to select between transformation methods via a button.
3. **Visualization**:
   - Display original and aligned points in different colors.
   - Visualize transformed points with movement as lines.
   - Show reconstructed transformation and scale parameters in text.
