# Environment-Invariant Curriculum Relation Learning for Fine-Grained Scene Graph Generation in Pytorch


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions, the recommended configuration is cuda-10.1 & pytorch-1.7.1.  

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing (VG & GQA).

## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxT9s3JwIpoGz4cA?e=usU6TR). For GQA dataset, we pretrained a new object detector, you can get it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxBfihou2smfXFV9?e=VtyoR7). However, we recommend you to pretrain a new one on GQA since we do not pretrain it for multiple times to choose the best pre-trained model for extracting offline region-level features.

## Perform training on Scene Graph Generation

### Set the dataset path

First,  organize all the files like this:
```bash
datasets
  |-- vg
    |--detector_model
      |--pretrained_faster_rcnn
        |--model_final.pth
      |--GQA
        |--model_final_from_vg.pth       
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    
  |--gqa
    |--images
      |--.... (images)
    |--GQA_200_ID_Info.json
    |--GQA_200_Train.json
    |--GQA_200_Test.json
```

### Choose a dataset

You can choose the training/testing dataset by setting the following parameter:
``` bash
GLOBAL_SETTING.DATASET_CHOICE 'VG'  #['VG', 'GQA']
```

### Choose a task

To comprehensively evaluate the performance, we follow three conventional tasks: 1) **Predicate Classification (PredCls)** predicts the relationships of all the pairwise objects by employing the given ground-truth bounding boxes and classes; 2) **Scene Graph Classification (SGCls)** predicts the objects classes and their pairwise relationships by employing the given ground-truth object bounding boxes; and 3) **Scene Graph Detection (SGDet)** detects all the objects in an image, and predicts their bounding boxes, classes, and pairwise relationships.

For **Predicate Classification (PredCls)**, you need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Choose your model

We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. You can use ```GLOBAL_SETTING.RELATION_PREDICTOR``` to select one of them:

```bash
GLOBAL_SETTING.RELATION_PREDICTOR 'EICR_model'
```


The default settings are under ```configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```.

### Choose your Encoder 

You need to further choose an object/relation encoder for "Motifs" or "VCTree" or "Self-Attention" predictor, by setting the following parameter:

```bash
GLOBAL_SETTING.BASIC_ENCODER 'Motifs'
```



### Examples of the Training Command
Training Example 1 : (VG, Motifs, PredCls)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10050 --nproc_per_node=1 ./tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'EICR_model' GLOBAL_SETTING.BASIC_ENCODER 'Motifs' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 120000 SOLVER.VAL_PERIOD 10000 SOLVER.CHECKPOINT_PERIOD 10000 GLOVE_DIR /data/myk/newreason/SHA/datasets/vg OUTPUT_DIR /data/myk/newreason/ICCV23/SHA/output/VG_predcls_EICR SOLVER.SCHEDULE.TYPE WarmupMultiStepLR    SOLVER.STEPS "(56000, 96000)"
```

Training Example 2 : (GQA_200, Motifs, SGCls)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10050 --nproc_per_node=1 ./tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'GQA_200' GLOBAL_SETTING.RELATION_PREDICTOR 'EICR_model' GLOBAL_SETTING.BASIC_ENCODER 'Motifs' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 120000 SOLVER.VAL_PERIOD 10000 SOLVER.CHECKPOINT_PERIOD 10000 GLOVE_DIR /data/myk/newreason/SHA/datasets/vg OUTPUT_DIR /data/myk/newreason/ICCV23/SHA/output/VG_predcls_EICR SOLVER.SCHEDULE.TYPE WarmupMultiStepLR    SOLVER.STEPS "(56000, 96000)"
```

## Evaluation

You can evaluate it by running the following command.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10083 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'GQA_200' GLOBAL_SETTING.RELATION_PREDICTOR 'EICR_model' GLOBAL_SETTING.BASIC_ENCODER 'Motifs' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/myk/home/reason/newreason/SHA/datasets/vg/glove OUTPUT_DIR /home/myk/home/reason/newreason/SHA/output/GQA_precl_motif3samples_09aplha_start30000end60000/
```



## Citation
```bash
@inproceedings{min2023environment,
  title={Environment-Invariant Curriculum Relation Learning for Fine-Grained Scene Graph Generation},
  author={Min, Yukuan and Wu, Aming and Deng, Cheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13296--13307},
  year={2023}
}
```



## Acknowledgment

Our code is on top of [SHA-GCL-for-SGG](https://github.com/dongxingning/SHA-GCL-for-SGG), we sincerely thank them for their well-designed codebase.
