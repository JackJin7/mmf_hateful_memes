# Hateful Memes Challenge using MMF

This is a project for the course CSC2516 in U of T. The code is based on MMF developed by facebook AI and is forked from the MMF project [example](https://github.com/apsdehal/hm_example_mmf). 

## Prerequisites

Please follow prerequisites for the Hateful Memes dataset at [this link](https://fb.me/hm_prerequisites).

## Running

Training ViLBERT or Visual BERT model with the following command on the Hateful Memes dataset:

```
# unimodally pretrained
MMF_USER_DIR="." mmf_run config="configs/experiments/vilbert/defaults.yaml" model=my_vilbert dataset=hateful_memes run_type=train_val
MMF_USER_DIR="." mmf_run config="configs/experiments/visual_bert/defaults.yaml" model=my_visual_bert dataset=hateful_memes run_type=train_val

# multimodally pretrained
MMF_USER_DIR="." mmf_run config="configs/experiments/vilbert/from_cc.yaml" model=my_vilbert dataset=hateful_memes run_type=train_val
MMF_USER_DIR="." mmf_run config="configs/experiments/visual_bert/from_coco.yaml" model=my_visual_bert dataset=hateful_memes run_type=train_val
```

Evaluating the model:

```
# unimodally pretrained
MMF_USER_DIR="." mmf_run config="configs/experiments/vilbert/defaults.yaml" model=my_vilbert dataset=hateful_memes run_type=val_test checkpoint.resume_file=[YOUR_MODEL_PATH] checkpoint.resume_pretrained=False
MMF_USER_DIR="." mmf_run config="configs/experiments/visual_bert/defaults.yaml" model=my_visual_bert dataset=hateful_memes run_type=val_test checkpoint.resume_file=[YOUR_MODEL_PATH] checkpoint.resume_pretrained=False

# multimodally pretrained
MMF_USER_DIR="." mmf_run config="configs/experiments/vilbert/from_cc.yaml" model=my_vilbert dataset=hateful_memes run_type=val_test checkpoint.resume_file=[YOUR_MODEL_PATH] checkpoint.resume_pretrained=False
MMF_USER_DIR="." mmf_run config="configs/experiments/visual_bert/from_coco.yaml" model=my_visual_bert dataset=hateful_memes run_type=val_test checkpoint.resume_file=[YOUR_MODEL_PATH] checkpoint.resume_pretrained=False
```



