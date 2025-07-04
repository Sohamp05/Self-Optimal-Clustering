# Self Optimal Clustering (SOC)

This repository contains the complete MATLAB implementation of the Self Optimal Clustering (SOC) algorithm for image segmentation, along with baseline methods IMC-1 and IMC-2.

The project is part of an academic course submission and includes experimental results, segmentation visualizations, and metric-based comparisons across clustering techniques.

## Overview

SOC improves upon traditional mountain clustering by dynamically optimizing the clustering threshold using silhouette analysis and Lagrange interpolation. This helps SOC adapt to different types of image data more effectively than fixed-threshold methods like IMC-1 and IMC-2.


## How to Use

1. Clone the repository and open MATLAB.
2. Add the folder to your MATLAB path.
3. Place any RGB image in the `images/` folder.
4. Modify `soc_implmt.m` to select your image and number of clusters.
5. Run the script. Segmented images and GSI scores will be saved in `results/`.

Example snippet inside `main_script.m`:
```matlab
a = imread('input/sample.jpg');
nk = 4;
x = double(reshape(a, [], 3));

fac = factorcal(x, nk, 1);
resSOC = soc(x, nk, fac);
resIMC1 = imc1(x, nk, delta1);
resIMC2 = imc2(x, nk, delta1, delta2);

% Visualization and GSI display handled automatically
