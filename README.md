# SAR-VQA
Here is the implementation of our ACL-2021 [Check It Again: Progressive Visual Question Answering via Visual Entailment](https://arxiv.org/).
This repository contains code modified from [here for SAR+SSL](https://github.com/CrossmodalGroup/SSL-VQA) and [here for SAR+LMH](https://github.com/chrisc36/bottom-up-attention-vqa), many thanks!
## Requirements
* python 3.7.6
* pytorch 1.5.0
* zarr
* tdqm
* spacy
* h5py

## Download and preprocess the data
```Bash
 cd data 
 bash download.sh
 python preprocess_image.py --data trainval
 python create_dictionary.py --dataroot vqacp2/
 python preprocess_text.py --dataroot vqacp2/ --version v2
 cd ..
```

## Train Candidate Answers Selector and Build The Datasets for The Answers Re-ranking Module
* The VQA model applied as CAS is free choice in our framework. In this paper, we mainly use SSL as CAS. 


* The setting of model training of CAS can be refered in [SSL](https://github.com/CrossmodalGroup/SSL-VQA). 


* To build the Dataset for the Answer Re-ranking module based on Visual Entailment, we modified the SSL's code of `VQAFeatureDataset()` in [dataset_vqacp.py](https://github.com/CrossmodalGroup/SSL-VQA/blob/master/dataset_vqacp.py) and `evaluate()` in [train.py](https://github.com/CrossmodalGroup/SSL-VQA/blob/master/train.py).  The modified codes are avaliable in `CAS_scripts`, just replace the corresponding class/function in [SSL](https://github.com/CrossmodalGroup/SSL-VQA).


* After the Candidate Answers Selecting Module, we can get `TrainingSet_top20_candidates.json` and `TestSet_top20_candidates.json` files as the training and test set for Answer Re-ranking Module,respectively.

## Training( Answer Re-ranking based on Visual Entailment)
* Train Top12-SAR
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_main.py --output saved_models_cp2/ --lp 0 --train_condi_ans_num 12
```
* Train Top20-SAR
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_main.py --output saved_models_cp2/ --lp 0 --train_condi_ans_num 20
```
* Train Top12-SAR+SSL
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_main.py --output saved_models_cp2/ --lp 1 --self_loss_weight 3 --train_condi_ans_num 12
```
* Train Top20-SAR+SSL
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_main.py --output saved_models_cp2/ --lp 1 --self_loss_weight 3 --train_condi_ans_num 20
```
* Train Top12-SAR+LMH
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_main.py --output saved_models_cp2/ --lp 2  --train_condi_ans_num 12
```
* Train Top20-SAR+LMH
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_main.py --output saved_models_cp2/ --lp 2  --train_condi_ans_num 20
```



