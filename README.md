# Relative Pose Estimation from Plane Correspondences

This is part of the code for my semester project at ETH Zurich. The goal of this project is to estimate the relative pose between two cameras from a set of plane correspondences. 

![pipeline](images/Full%20Pipeline-Full.drawio.png)

![vis](images/vis_pipeline.png)

## Dependencies
To generate image pairs from Hypersim Dataset, please install Open3D

To run the pipeline, please install the latest version of Pytorch, OpenCv, and Kornia

## Model Weights
The trained model for each stage can be downloaded from [here](https://drive.google.com/drive/folders/1iVoUJGaS1G5IWKXZxCsw1bs_1PEuSRCf?usp=share_link)

## Generate image pairs
To generate image pairs, first download the [Hypersim Dataset](https://github.com/apple/ml-hypersim).
Then you can generate image pairs using `Dataset/generate_image_pairs.py`.


## Run the Pipeline
First download the model weights and change the path in `raftba.py`
then run the following command
```
python raftba.py
```
