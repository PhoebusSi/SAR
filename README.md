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

## Train Candidate Answers Selector & Build the datasets for the Answers Re-ranking module
* The VQA model applied as Candidate Answer Selector(CAS) is a free choice in our framework. In this paper, we mainly use SSL as CAS. 


* The setting of model training of CAS can be refered in [SSL](https://github.com/CrossmodalGroup/SSL-VQA). 


* To build the Dataset for the Answer Re-ranking module based on Visual Entailment, we modified the SSL's code of `VQAFeatureDataset()` in [dataset_vqacp.py](https://github.com/CrossmodalGroup/SSL-VQA/blob/master/dataset_vqacp.py) and `evaluate()` in [train.py](https://github.com/CrossmodalGroup/SSL-VQA/blob/master/train.py).  The modified codes are avaliable in `CAS_scripts`, just replace the corresponding class/function in [SSL](https://github.com/CrossmodalGroup/SSL-VQA).


* After the Candidate Answers Selecting Module, we can get `train_top20_candidates.json` and `test_top20_candidates.json` files as the training and test set for Answer Re-ranking Module,respectively.

## Builed Top20-Candidate-Answers dataset (entries) for training/test the model of Answer Re-ranking module
If you don't want to train CAS model(e.g. SSL) to build the datasets in the way mentioned above, you can download the rebuiled top20-candidate-answers dataset (with different Qiestion-Answer-Combination strategies) from [here]('www.baidu.com'). 

* Put the downloaded Pickle file into the `data4VE` folder, then the code will load and rebuild it into the `entries` which will be feed in `__getitem__()` of dataloader. (Skipping all data processing steps of the Answer Re-ranking based on Visual Entailment directly)
* Each entry of the entries rebuiled from this Pickle file includes `image_features`, `image_spatials`, `top20_score`, `question_id`, `QA_text_ids`, `top20_label`, `answer_type`, `question_text`, `LMH_bias`, where `QA_text_ids` is the question-answer-combination(R/C) ids obtained from the LXMERT tokenizer. 


## Training (Answer Re-ranking based on Visual Entailment)
* Train Top12-SAR
```Bash
CUDA_VISIBLE_DEVICES=0,1 python SAR_main.py --output saved_models_cp2/ --lp 0 --train_condi_ans_num 12
```
* Train Top20-SAR
```Bash
CUDA_VISIBLE_DEVICES=0,1 python SAR_main.py --output saved_models_cp2/ --lp 0 --train_condi_ans_num 20
```
* Train Top12-SAR+SSL
```Bash
CUDA_VISIBLE_DEVICES=0,1 python SAR_main.py --output saved_models_cp2/ --lp 1 --self_loss_weight 3 --train_condi_ans_num 12
```
* Train Top20-SAR+SSL
```Bash
CUDA_VISIBLE_DEVICES=0,1 python SAR_main.py --output saved_models_cp2/ --lp 1 --self_loss_weight 3 --train_condi_ans_num 20
```
* Train Top12-SAR+LMH
```Bash
CUDA_VISIBLE_DEVICES=0,1 python SAR_main.py --output saved_models_cp2/ --lp 2  --train_condi_ans_num 12
```
* Train Top20-SAR+LMH
```Bash
CUDA_VISIBLE_DEVICES=0,1 python SAR_main.py --output saved_models_cp2/ --lp 2  --train_condi_ans_num 20
```


The function `evaluate()` in `SAR_train.py` is used to select the best model during training, without QTD module yet. The trained QTD model is used in `SAR_test.py` where we obtain the final test score.  

## Evaluation
* Evaluate trained SAR model
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_test.py  --checkpoint_path4test saved_models_cp2/SAR_top12_best_model.pth --output saved_models_cp2/result/ --lp 0 --QTD_N4yesno 1 --QTD_N4non_yesno 12
```
* Evaluate trained SAR+SSL model
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_test.py  --checkpoint_path4test saved_models_cp2/SAR_SSL_top12_best_model.pth --output saved_models_cp2/result/ --lp 1 --QTD_N4yesno 1 --QTD_N4non_yesno 12
```
* Evaluate trained SAR+LMH model
```Bash
CUDA_VISIBLE_DEVICES=0 python SAR_test.py  --checkpoint_path4test saved_models_cp2/SAR_LMH_top12_best_model.pth --output saved_models_cp2/result/ --lp 2 --QTD_N4yesno 2 --QTD_N4non_yesno 12
```
* Note that we mainly use `R->C` Question-Answer Combination Strategy, which can always achieves or rivals the best performance on SAR/SAR+SSL/SAR+LMH. Specifically, we ﬁrst use strategy `R` ( `SAR_replace_dataset_vqacp.py`) at training, which is aimed at preventing the model from excessively focusing on the co-occurrence relation between question category and answer, and then use strategy `C`(`SAR_concatenate_dataset_vqacp.py`) at testing to introduce more information for inference. 
* Compute detailed accuracy for each answer type:
```bash
python comput_score.py --input saved_models_cp2/result/XX.json --dataroot data/vqacp2/cache
```

