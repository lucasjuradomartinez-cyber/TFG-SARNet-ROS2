# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 

# unlabeled, first-responder, civilian, vegetation, road, dirt-road, building, sky, civilian-car, responder-vehicle, debris, command-post
def get_palette():
    unlabeled          = [0,0,0]
    first_responder    = [255,130,0]
    civilian           = [224,67,217]
    vegetation         = [65,117,5]
    road               = [155,155,155]
    dirt_road          = [255,221,104]
    building           = [180,117,31]
    sky                = [22,152,207]
    civilian_car       = [64,0,128]
    responder_vehicle  = [192,64,0]
    debris             = [60,37,0]
    command_post       = [192,128,128]
    palette    = np.array([unlabeled, first_responder, civilian, vegetation, road, dirt_road, building, sky, civilian_car, responder_vehicle, debris, command_post])
    return palette

def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('./SSRR_video/result/Pred/'+weight_name + '_' + image_name[i] + '.jpg')
        #img.save('./visual_label/' + image_name[i])

def visualize_v2(image_name, predictions, weight_name):
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy().astype(np.uint8)  # Asegurar tipo uint8
        img = Image.fromarray(pred)  # Convertir directamente a imagen en escala de grises
        img.save(f'./SSRR_video/result/Pred/SARNet_CBAM_V2/{weight_name}_{image_name[i]}.png')



def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
