import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython import display

image_file = json.load(open('/data/myk/newreason/SHA/datasets/vg/image_data.json'))
vocab_file = json.load(open('/data/myk/newreason/SHA/datasets/vg/VG-SGG-dicts-with-attri.json'))
data_file = h5py.File("/data/myk/newreason/SHA/datasets/vg/VG-SGG-with-attri.h5", 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

# load detected results
#detected_origin_path = '/data/myk/newreason/SHA/output/VG_precl_motif/inference/VG_stanford_filtered_with_attribute_test/'

#############model path##################
detected_origin_path = '/data/myk/newreason/SHA/output/VG_precls_motif3samples_09alpha_normandover_start30000end60000/inference/VG_stanford_filtered_with_attribute_test/'

detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))

#get image info by index
def get_info_by_idx(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    # boxes
    boxes = groundtruth.bbox
    # object labels
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]
    pred_labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    #mask = pred_rel_score > thres
    #pred_rel_score = pred_rel_score[mask]
    #pred_rel_label = pred_rel_label[mask]
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    return img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label,idx


def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)


def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))

def print_list_txt(name, input_list,f):
    for i, item in enumerate(input_list):
        f.write(name + ' ' + str(i) + ': ' + str(item)+'\n')


def draw_image(img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label,idx, print_img=True):
    pic = Image.open(img_path)

    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    copy_image = pic.copy()
    ################save#################
    copy_image.save('/data/myk/newreason/SHA/visualiza_result/EICR/imgs/%d.jpg' % (idx))
    if print_img:
        #display(pic)
        #pic.show()
        print(idx)
        # plt.figure(img_path)
        # imshow(pic)
        # plt.plot()
    #if print_img:


        # print('*' * 50)
        # print_list('gt_boxes', labels)
        # print('*' * 50)
        # print_list('gt_rels', gt_rels)
        # print('*' * 50)
    ################save#################
    f = open('/data/myk/newreason/SHA/visualiza_result/EICR/relation/ouput_%d.txt' % (idx), 'w')




    print_list_txt('pred_rels', pred_rels[:20],f)
    #print('*' * 50)
    f.close()
    return None


# %%
def show_selected(idx_list):
    for select_idx in idx_list:
        print(select_idx)
        draw_image(*get_info_by_idx(select_idx, detected_origin_result))


def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx + length):
        print(cand_idx)
        draw_image(*get_info_by_idx(cand_idx, detected_origin_result))


# %%
#show_all(start_idx=0, length=5)
#show_selected([119, 967, 713, 5224, 19681, 25371])
for i in range(1000):
    show_selected([i])