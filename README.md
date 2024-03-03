# TESI MAGISTRALE

<!-- <details>
<summary>
    <b>TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification</b>. <a href="https://proceedings.neurips.cc/paper/2021/file/10c272d06794d3e5785d5e7c5356e9ff-Paper.pdf" target="blank">[NeurIPS2021]</a>
</summary>

```tex
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
```

**Abstract:** With the development of computational pathology, deep learning methods for Gleason grading through whole slide images (WSIs) have excellent prospects. Since the size of WSIs is extremely large, the image label usually contains only slide-level label or limited pixel-level labels. The current mainstream approach adopts multi-instance learning to predict Gleason grades. However, some methods only considering the slide-level label ignore the limited pixel-level labels containing rich local information. Furthermore, the method of additionally considering the pixel-level labels ignores the inaccuracy of pixel-level labels. To address these problems, we propose a mixed supervision Transformer based on the multiple instance learning framework. The model utilizes both slidelevel label and instance-level labels to achieve more accurate Gleason grading at the slide level. The impact of inaccurate instance-level labels is further reduced by introducing an eﬃcient random masking strategy in the mixed supervision training process. We achieve the state-of-the-art performance on the SICAPv2 dataset, and the visual analysis shows the accurate prediction results of instance level.

</details>

![overview](docs/overview.png) -->

<!-- ## Data Preprocess

we follow the CLAM's WSI processing solution (https://github.com/mahmoodlab/CLAM)

```bash
# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch

# Feature Extraction
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
``` -->

## Installation

- Windows (Tested on Windows 11)
- CPU (Tested on CPU Intel Core i7 8th Gen)
- Python (3.7.16)

Please refer to the following instructions:

```bash
# create and activate the conda environment
conda create -n yourenvname python==3.7.16 -y
conda activate yourenvname
pip install -r requirements.txt
```

<!-- # install pytorch
## pip install
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
## conda install
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install related package
pip install -r requirements.txt
``` -->

### Train

#### mi-net (https://arxiv.org/abs/1610.02501):
```python
python main.py test_description --model=minet --dataset=elephant
python main.py test_description --model=minet --dataset=fox
python main.py test_description --model=minet --dataset=tiger
python main.py test_description --model=minet --dataset=musk1
python main.py test_description --model=minet --dataset=musk2
python main.py test_description --model=minet --dataset=newsgroups
python main.py test_description --model=minet --dataset=messidor
```
#### MI-net (https://arxiv.org/abs/1610.02501):
```python
python main.py test_description --model=MInet --dataset=elephant
python main.py test_description --model=MInet --dataset=fox
python main.py test_description --model=MInet --dataset=tiger
python main.py test_description --model=MInet --dataset=musk1
python main.py test_description --model=MInet --dataset=musk2
python main.py test_description --model=MInet --dataset=newsgroups
python main.py test_description --model=MInet --dataset=messidor
```
#### Att-net (https://arxiv.org/abs/1802.04712):
```python
python main.py test_description --model=attnet --dataset=elephant
python main.py test_description --model=attnet --dataset=fox
python main.py test_description --model=attnet --dataset=tiger
python main.py test_description --model=attnet --dataset=musk1
python main.py test_description --model=attnet --dataset=musk2
python main.py test_description --model=attnet --dataset=newsgroups
python main.py test_description --model=attnet --dataset=messidor
```
##### Default values:
```python
--run=5
--folds=10
--epochs=50
--no-cuda=True
--config=settings.yaml #where you can change hyperparameteres
--eval-per-epoch=0 #Choose 0 if you do not want to save the best model, otherwise choose the number of times per epoch you want to save the best model (based on test set)
```

<!-- ### Test

```python
python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=0
```

## Reference

- If you found our work useful in your research, please consider citing our works(s) at:

```tex

@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}


```

© This code is made available under the GPLv3 License and is available for non-commercial academic purposes. -->
