# Color constancy from a pure color view

Shuwei Yue and *[Minchen Wei](https://www.polyucolorlab.com/)

*Color, Imaging, and Metaverse Research Center, The Hong Kong Polytechnic University.*

![PCC-MLP-model](https://user-images.githubusercontent.com/106613332/212873202-a61d95e0-7e59-4c1c-8307-1e0ff74038bd.png)
If you use this code, please cite our paper:

```
@article{yue2023color,
  title={Color constancy from a pure color view},
  author={Yue, Shuwei and Wei, Minchen},
  journal={JOSA A},
  volume={40},
  number={3},
  pages={602--610},
  year={2023},
  publisher={Optica Publishing Group}
}

```

## Code

####  Prerequisite   

* Pytorch
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
---
> Update on Sep. 29, 2024, There are some differences between the two formats:

- The **Thumbnail** version has `raw` data in `png` format, and the `label` is in `json` format, which includes extracted information from a 24-color card.
- The **Full-resolution** version has both `raw` and `label` in `npy` format, where the `label` contains only RGB information of the light source.

It is recommended to directly use the `full-resolution` version for training.

---

So, it is better to preprocess your data and corresponding labels into *.npy* format.
The processed *Recommend-ColorChecker dataset* (**CC2018**) is provided in the folder of `datasets/CC2018/`. Noted that the black level and masked card of this dataset have been subtracted, then resized to $64\times64$ with normalized.

- step 2:

Run `train.py`  

>  You can change the training fold number in `config/param_config.py`  

#### Testing

move your trained model from `log` folder to the `pretrain_models`

Run `test.py`

## PolyU Pure Color dataset

Thumbnails(`.png`) and full-resolution images(`.npy`) are provided in the following link:

 [PolyU PureColor dataset](https://www.123865.com/s/eNurTd-oFWmH)
 
 ![thumbnails-view](https://user-images.githubusercontent.com/106613332/212550111-52245d3b-f989-4cf8-8fcf-82187d46cbfc.png)

