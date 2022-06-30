<p align="center">
  <img src = "./docs/imgs/readme_logo.png" width="60%">
</p>

`KAIST-Radar (K-Radar)` (provided by ['AVELab'](http://ave.kaist.ac.kr/)) is a novel large-scale object detection dataset and benchmark that contains 35K frames of 4D Radar tensor (4DRT) data with power measurements along the Doppler, range, azimuth, and elevation dimensions, together with carefully annotated 3D bounding box labels of objects on the roads. K-Radar includes challenging driving conditions such as adverse weathers (fog, rain, and snow) on various road structures (urban, suburban roads, alleyways, and highways). In addition to the 4DRT, we provide auxiliary measurements from carefully calibrated high-resolution Lidars, surround stereo cameras, and RTK-GPS. This repository provides the `K-Radar` dataset, annotation tool for 3d bounding boxes, and the visualization tool for showing the inference results and calibrating the sensors.

![image](./docs/imgs/kradar_examples.png)

The URLs listed below are useful for understanding the K-Radar dataset and benchmark:
* <a href="https://arxiv.org/abs/2206.08171"> K-Radar paper and appendix </a>
* <a href="http://youtube.com/watch?v=XWAi71AUo5A&feature=youtu.be"> The video clip that shows each sensor measurement dynamically changing during driving under the heavy snow condition </a>
* <a href="https://www.youtube.com/watch?v=DD3Iks8sB3I"> The video clip that shows the 4D radar tensor & Lidar point cloud (LPC) calibration and annotation process </a>
* <a href="https://www.youtube.com/watch?v=KyDI32Uck4g"> The video clip that shows the annotation process in the absence of LPC measurements of objects  </a>
* <a href="https://www.youtube.com/watch?v=q0scC3KZlyI"> The video clip that shows calibration results </a>
* <a href="https://www.youtube.com/watch?v=b_9TtOxaN1w"> The video clip that shows the GUI-based program for visualization and neural network inference </a>

# K-Radar Dataset
This is the documentation for how to use our detection frameworks with K-Radar dataset.
We tested the K-Radar detection frameworks on the following environment:
* Python 3.8
* Ubuntu 18.04/20.04
* Torch 1.9.1
* CUDA 11.2

## Preparing the Dataset
1. To download the dataset, log in to <a href="https://kaistavelab.direct.quickconnect.to:54568/"> our server </a> with the following credentials: 
      ID       : kradards
      Password : (We are preparing the server to make K-Radar dataset public.)
2. Go to the "File Station" folder, and download the dataset by right-click --> download.
   Note for Ubuntu user, there might be some error when unzipping the files. Please check the "readme_to_unzip_file_in_linux_system.txt".
3. After all files are downloaded, please arrange the workspace directory with the following structure:
```
KRadarFrameworks
├── annot_calib_tools
├── devkits
      ├── configs
      ├── dataset_utils
      ├── datasets
      ├── models
      ├── pipelines
      ├── resources
      ├── uis
      ├── utils
├── kradar
      ├── kradar_0
            ├── 1
            ├── 2
            ...
      ├── kradar_1
            ...
            ├── 57
            ├── 58
├── logs
```
