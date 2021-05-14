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

## Training Candidate Answers Selector
The VQA model applied as CAS is free choice in our framework. In this paper, we mainly use SSL as CAS. 

