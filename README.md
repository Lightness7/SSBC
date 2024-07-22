# SSBC
For the hyperspectral image (HSI) classification task, we propose the method for extracting **spectral-spatial-band correlation (SSBC) features**. 

## Introduction
To address the over-smoothing problem in classification maps, we design the method for extracting SSBC features, which fuse the **joint spectral-spatial features**, original spectral bands and **band correlation features**. In this study, the classification maps generated by the **joint spectral-spatial features** suffer from the significant over-smoothing problem, which is manifested in significant differences from the original image in terms of the object boundaries and details. Here we argue that increasing spectral information in the extracted features is the key to addressing this problem. Thus, both the original spectral bands and **band correlation features** are fused into the **joint spectral-spatial features** as the added spectral information. The SSBC features can efficiently mitigate the over-smoothing problem in the classification maps.

- The extraction of the **joint spectral-spatial features** is considered as a discrete cosine transform (DCT)-based information compression, where a flattening operation is used to avoid the high computational cost caused by the fact that the joint spectral-spatial information generally requires to be distilled from 3D images.
- The extraction of the **band correlation features** improves the calculations of normalized difference vegetation index (NDVI) and ironoxide (IO) since their calculations involving two spectral bands are not appropriate for the abundant spectral bands of HSI.

## Environment
- Python 3.10.9
- Numpy 1.21.5
- Pandas 1.5.3
- OpenCV 4.6.0
- Scikit-learn 1.1.2
- Imblearn 0.0
- Lightgbm 3.3.5
- Shap 0.41.0
- Matplotlib 3.7.1
- Alive_progress 3.1.4

It is suggested to use **Anaconda** for configuring the Python environment and then **PyCharm** for running the demo.

## Implementation
Please run

```python
SSBC_demo.py
```

In line 396-415 of **SSBC_demo.py**, there are some parameters that require to be set:

- **dataset_name**: Choose a HSI dataset you wish;
- **original_spectral**: Whether to involve original spectral bands in the features;
- **class_map_output**: Whether to generate a classification map, implemented only if **experiment_time** is set to **1**;
- **shap_output**: Whether to generate a summary plot of SHAP, implemented only if **experiment_time** is set to **1**;
- **experiment_time**: 1 experiment or the average of 10 experiments;
- **window_size**: Size of local window;
- **r**: Retain the first *r* rows of the low-frequency components in the matrix after DCT, ranging from 1 to **window_size**;
- **band_range**: Number of groups in the calculation of the band correlation features;
- **test_size**: Proportion of test set divided.

where **window_size**, **r** and **band_range** are the key parameters in terms of performance.





