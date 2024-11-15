# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import bisect
import copy
import logging

import json
import torch
import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms

# by Jiaxin
def get_dataset_statistics(cfg):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-'*100)
    logger.info('get dataset statistics...')
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        dataset_names = cfg.DATASETS.VG_TRAIN
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        dataset_names = cfg.DATASETS.GQA_200_TRAIN
    else:
        dataset_names = None
        exit('wrong Dataset name!')

    data_statistics_name = ''.join(dataset_names) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
    
    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-'*100)
        return torch.load(save_file, map_location=torch.device("cpu"))

    statistics = []
    for dataset_name in dataset_names:
        data = DatasetCatalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        dataset = factory(**args)
        statistics.append(dataset.get_statistics())
    logger.info('finish')

    assert len(statistics) == 1
    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        result = {
            'fg_matrix': statistics[0]['fg_matrix'],
            'pred_dist': statistics[0]['pred_dist'],
            'obj_classes': statistics[0]['obj_classes'], # must be exactly same for multiple datasets
            'rel_classes': statistics[0]['rel_classes'],
            'stat_classes':statistics[0]['stat_classes']
        }

    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        result = {
            'fg_matrix': statistics[0]['fg_matrix'],
            'pred_dist': statistics[0]['pred_dist'],
            'obj_classes': statistics[0]['obj_classes'], # must be exactly same for multiple datasets
            'rel_classes': statistics[0]['rel_classes'],
        }

    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-'*100)
    torch.save(result, save_file)
    return result


def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:#按长宽比分组
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)#将长宽比list转化为group_id list。[0,1,1,0,1,0,0,0,1]这样的
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0):
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'

    ##################train_data
    #is_train=False
    ####################



    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        if mode == 'train':
            dataset_list = cfg.DATASETS.VG_TRAIN
        elif mode == 'val':
            dataset_list = cfg.DATASETS.VG_VAL
        else:
            dataset_list = cfg.DATASETS.VG_TEST
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        if mode == 'train':
            dataset_list = cfg.DATASETS.GQA_200_TRAIN
        elif mode == 'val':
            dataset_list = cfg.DATASETS.GQA_200_VAL
        else:
            dataset_list = cfg.DATASETS.GQA_200_TEST
    else:
        dataset_list = None
        exit('wrong dataset choice!')

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    data_loaders = []
    #print(datasets[0].ind_to_classes)

    f = open("VGclasses.txt", "w")


    for line in datasets[0].ind_to_classes:
        f.write(line + '\n')
    f.close()


    ##########统计关系个数 ########数据形式ndarray
    relation_count=[]
    relation_stat={}
    relation_count_stat={}
    relation_context_stat={}
    relation_context_stat_sub = {}
    relation_context_stat_obj = {}
    obj_stat={}
    data_idx=0
    context_stat = {}
    for relation in datasets[0].relationships:
        relation_count.append(len(relation))
        relation_label=relation[:,-1]

        subject_index = relation[:, 0]
        object_index = relation[:, 1]
        for sub,obj,rel in zip(subject_index,object_index,relation_label):
            subject_label=datasets[0].gt_classes[data_idx][sub]
            object_label = datasets[0].gt_classes[data_idx][obj]
            subject_text = datasets[0].ind_to_classes[subject_label]
            object_text = datasets[0].ind_to_classes[object_label]
            context=str(subject_text+"-"+object_text)
            if context not in context_stat:
                context_stat[context] = 1
            else:
                context_stat[context] += 1
            if datasets[0].ind_to_predicates[rel] not in relation_context_stat:
                relation_context_stat[datasets[0].ind_to_predicates[rel]] = {}
            else:
                if context not in relation_context_stat[datasets[0].ind_to_predicates[rel]]:
                    relation_context_stat[datasets[0].ind_to_predicates[rel]][context] = 1
                else:
                    relation_context_stat[datasets[0].ind_to_predicates[rel]][context] += 1

            if datasets[0].ind_to_predicates[rel] not in relation_context_stat_sub:
                relation_context_stat_sub[datasets[0].ind_to_predicates[rel]] = {}
            else:
                if subject_text not in relation_context_stat_sub[datasets[0].ind_to_predicates[rel]]:
                    relation_context_stat_sub[datasets[0].ind_to_predicates[rel]][subject_text] = 1
                else:
                    relation_context_stat_sub[datasets[0].ind_to_predicates[rel]][subject_text] += 1


            if datasets[0].ind_to_predicates[rel] not in relation_context_stat_obj:
                relation_context_stat_obj[datasets[0].ind_to_predicates[rel]] = {}
            else:
                if object_text not in relation_context_stat_obj[datasets[0].ind_to_predicates[rel]]:
                    relation_context_stat_obj[datasets[0].ind_to_predicates[rel]][object_text] = 1
                else:
                    relation_context_stat_obj[datasets[0].ind_to_predicates[rel]][object_text] += 1



        for r in relation_label:
            if datasets[0].ind_to_predicates[r] not in relation_count_stat:
                relation_count_stat[datasets[0].ind_to_predicates[r]] = 1
            else:
                relation_count_stat[datasets[0].ind_to_predicates[r]] += 1
        #print(str(len(relation)) in relation_stat)
        if len(relation) not in relation_stat:
            relation_stat[len(relation)] = 1
        else:
            relation_stat[len(relation)] += 1
        data_idx+=1

    for objects in datasets[0].gt_classes:
        for i in objects:
            if datasets[0].ind_to_classes[i] not in obj_stat:
                obj_stat[datasets[0].ind_to_classes[i]]=1
            else:
                obj_stat[datasets[0].ind_to_classes[i]] += 1
    relation_keys=list(relation_context_stat.keys())
    rel_and={}
    rel_and_count=[]
    # for i in range(len(relation_keys)-1):
    #     key1=relation_context_stat[relation_keys[0]].keys()
    #     key2 = relation_context_stat[relation_keys[i+1]].keys()
    #     intersection = key1 & key2
    #     rel_and[i]=list(intersection)
    #     rel_and_count.append(len(rel_and[i])/len(list(key2)))
    #     aaaa=1






    import xlwt
    import xlrd
    # xlwt,xlrd是python将数据导入excel表格使用的库
    i = 0
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('test')
    for rel_name,count in relation_context_stat_sub['on'].items():
            i += 1
            ws.write(i, 0, rel_name)
            ws.write(i, 1, count)
            wb.save('./on_intra_class_sub_test.xls')


    # for rel_num,stat in relation_stat.items():
    #         i += 1
    #         ws.write(i, 1, stat)
    #         ws.write(i, 0, rel_num)
    #         wb.save('./relstat_1.xls')
    # sort_dict=sorted(relation_stat.items(),key=lambda s:s[0])
    #
    #
    # import numpy as np
    # maxcount=np.max(relation_count)
    #
    # wb1 = xlwt.Workbook()
    # # 添加一个表
    # ws1 = wb1.add_sheet('test1')
    # for obj_num,stat in obj_stat.items():
    #         i += 1
    #         ws1.write(i, 1, stat)
    #         ws1.write(i, 0, int(obj_num))
    #         wb1.save('./objstat_1.xls')




    #print(datasets[0].ind_to_predicates)
    f = open("VGrelationclasses.txt", "w")

    for line in datasets[0].ind_to_predicates:
        f.write(line + '\n')
    f.close()

    # for i in range(len(datasets[0].relationships)):
    #     #print(len(datasets[0].relationships[i]))
    #     if len(datasets[0].relationships[i])==1:
    #         print("#################################################",i)




    for dataset in datasets:
        # print('============')
        # print(len(dataset))
        # print(images_per_gpu)
        # print('============')
        #stat_result=dataset.get_statistics()


        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)#使用零填充图片右下角使得形状相同（分辨率为 cfg.DATALOADER.SIZE_DIVISIBILITY）
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        #dataset负责给数据，batch_sample决定dataset如何采样数据，collate_fn决定怎样把多张图片组成一个batch
        # the dataset information used for scene graph detection on customized images
        if cfg.TEST.CUSTUM_EVAL:
            custom_data_info = {}
            custom_data_info['idx_to_files'] = dataset.custom_files
            custom_data_info['ind_to_classes'] = dataset.ind_to_classes
            custom_data_info['ind_to_predicates'] = dataset.ind_to_predicates

            if not os.path.exists(cfg.DETECTED_SGG_DIR):
                os.makedirs(cfg.DETECTED_SGG_DIR)

            with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json'), 'w') as outfile:  
                json.dump(custom_data_info, outfile)
            print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_data_info.json')) + ' SAVED !')
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


    # #DenseCLIP
    #     return data_loaders[0],datasets[0].ind_to_classes
    # return data_loaders,datasets[0].ind_to_classes
