# EG3D inversion projector

This is an **unofficial** inversion code of  **[eg3d](https://github.com/NVlabs/eg3d)**.

<video src="./eg3d/out/00056_w_plus.mp4"></video>



## installation

Please see **[eg3d](https://github.com/NVlabs/eg3d)** official repo for eg3d installation.

先在eg3d中配环境

然后保证如下环境：

CUDA 11.3 

```
torch                        1.11.0
torchvision                  0.12.0
```
- `pip install wandb==0.12.18`
- `pip install lpips==0.1.4` 





## convert pkl to pth file (optional)

This step is used to load the parameters from pkl checkpoint and save them to a pth file, so that code modifications on eg3d can take effect.

```
git clone https://github.com/aurelianocyp/EG3D-projector.git # 克隆完成后需要把eg3d的networks文件夹中的5个模型放到这个项目的eg3d文件夹中的networks中
cd eg3d # 进入的项目内的eg3d文件夹
python convert_pkl_2_pth.py --outdir=convert_pkl_2_pth_out --trunc=0.7    --network_pkl=networks/ffhq512-128.pkl --network_pth=networks/ffhq512-128.pth --sample_mult=2
```

Validation results will be saved to `convert_pkl_2_pth_out`, please check it.

pth file will be saved to `networks/ffhq512-128.pth`.

## Data preparation

**Notice:** 

For **FFHQ** images, please follow the guidance of **[eg3d](https://github.com/NVlabs/eg3d)** to re-align the FFHQ dataset and extract camera parameters (camera parameters can be found in [dataset.json](https://drive.google.com/uc?id=14mzYD1DxUjh7BGgeWKgXtLHWwvr-he1Z)) for input image.

For **wild** images, please refer to this [script](https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/ffhq/preprocess_in_the_wild.py) that can preprocess in-the-wild images compatible with the FFHQ checkpoints.

In this repo,  please prepare the input image `image_id.png` and its camera parameters `image_id.npy`. (please see the examplar data in  `./eg3d/projector_test_data`)

这里的preparation必须要做，将projector_test_data视为<indir>就行，如何处理参考eg3d中的数据preparation。必须要处理，否则用只有npy和png的文件夹虽然不会报错，但是会渲染出黑视频

eg3d和eg3d-projector放在一个机器中就行

处理数据时可能用到的命令（在auto-tmp目录下运行）

cp -r EG3D-projector/eg3d/projector_test_data/ eg3d/dataset_preprocessing/ffhq/

cd eg3d/dataset_preprocessing/ffhq/

cp -r projector_test_data/ Deep3DFaceRecon_pytorch/

复制回命令cp -r eg3d/dataset_preprocessing/ffhq/projector_test_data/ EG3D-projector/eg3d/

得到新图片的npy文件的方式，将camera.json删减到只剩下该图片的一个中括号即[......],然后将删减后的json与如下程序放在同一文件夹运行即可得到npy文件：
```python
import json
import numpy as np
with open('dataset.json', 'r') as file:
    data = json.load(file)
numpy_array = np.array(data)
np.save('numpy_data.npy', numpy_array)
```

## pretrained model

The projector needs vgg16 for loss computation, you can download vgg16.pt from https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt and save it to `EG3D-projector/eg3d/networks`.



## w and w_plus projector

This complementation reproduces the w_projector and w_plus_projector based on the projector [scripts](https://github.com/danielroich/PTI/tree/main/training/projectors) in [PTI](https://github.com/danielroich/PTI).

For w_projector :

```
cd eg3d
python run_projector.py --outdir=projector_out --latent_space_type w  --network=networks/ffhq512-128.pkl --sample_mult=2  --image_path ./projector_test_data/00018.png --c_path ./projector_test_data/00018.npy
```

Results will be saved to `./eg3d/projector_out/00018_w`



For w_plus projector:

```
cd eg3d
python run_projector.py --outdir=projector_out --latent_space_type w_plus  --network=networks/ffhq512-128.pkl --sample_mult=2  --image_path ./projector_test_data/00018.png --c_path ./projector_test_data/00018.npy
```

Results will be saved to `./eg3d/projector_out/00018_w_plus`

## PTI projector
先安装环境：
- `pip install kiwisolver==1.4.0`

**Notice:** before you run the PTI, please run the w or w_plusprojector to get the ''first_inv'' latent code (both w and w_plus are OK). 

Then run:

```
cd eg3d/projector/PTI
python run_pti_single_image.py
```

This script will automatically read the images in `./eg3d/projector_test_data`, and find their pivot latent code in `./eg3d/projector_out`, then finetune the eg3d model.



Results will be saved to eg3d/projector/PTI/checkpoints, named as `./checkpoints/model_{train_id}_{image name}_{latent space type}.pth`



You can run the following code to gen video using the obtained checkpoints :

```
# 在内eg3d目录下
python gen_videos_from_given_latent_code.py --outdir=out --npy_path ./projector_out/00018_w_plus/00018_w_plus.npy   --network=./projector/PTI/checkpoints/{Your PTI ckpt}.pth --sample_mult=2
```

# Next3d

在Next3d的环境下运行，将next3d中的training_avatar_texture文件夹拷贝到eg3d文件夹中。

将next3d中legacy的内容复制到eg3d的legacy中
