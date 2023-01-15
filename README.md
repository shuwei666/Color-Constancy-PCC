# Color constancy from a pure color view

Shuwei Yue and *[Minchen Wei](https://www.polyucolorlab.com/)

*Color, Imaging, and Metaverse Research Center, The Hong Kong Polytechnic University.*
![PCC-MLP-model](https://user-images.githubusercontent.com/106613332/212549365-c7d0bd52-aa4f-4289-b315-0e4efef8a000.png)

## Code

####  Prerequisite   

* Pytorch 1.11.0+cu113
* opencv-python  

#### Training

- step 1: Preparing the dataset

To train PCC, training/validation data should have the following formatting:

```
 datasets/numpy_data/
	|1_8D5U5524.npy
	|2_8D5U5525.npy
	...
 datasets/numpy_labels/
 	|1_8D5U5524.npy
	|2_8D5U5525.npy
	...
```

So, it is better to preprocess your data and corresponding labels into *.npy* format.
The processed *Recommend-ColorChecker dataset* (**CC2018**) are provided in the folder of `datasets/CC2018/`. Noted that the black level and masked card of this dataset have subtracted, then resized to $64\times64$ with normalized.

- step 2

Run `train.py`  

>  You can change the training fold number in `config/param_config.py`  

#### Testing

move your trained model from `log` folder to the `pretrain_models`

Run `test.py`

## PolyU Pure Color dataset

We will release the full dataset soon. For now, only thumbnails are provided in the following link:

 [pcc_huawei_thumbnail](https://connectpolyu-my.sharepoint.com/:f:/g/personal/21064184r_connect_polyu_hk/EjKJSnalMRpMn0nm4fJoIeEB34G9oxnu8MW7SYu2iSvC7g?e=dvk5vr)
 
 ![thumbnails-view](https://user-images.githubusercontent.com/106613332/212550111-52245d3b-f989-4cf8-8fcf-82187d46cbfc.png)

