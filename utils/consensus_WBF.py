#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:30:57 2023

@author: Erik Valle - gal20@mails.tsinghua.edu.cn
"""

import numpy as np
import os
import math
from itertools import permutations, combinations

from utils.ensemble_boxes_wbf_test import weighted_boxes_fusion

def convert_darknet(txt_list):
    """
    :param txt_list: a list of all labels for an image in the Darknet format: "category center_x center_y bbox_width bbox_height confidence"
    :return boxes_list, labels_list, scores_list after conversion
    """
    m_boxes = []
    m_cat = []
    m_scores = []
    for j in range(len(txt_list)):
        if txt_list[j] == '':
            continue
        cate, x_c, y_c, bbox_width, bbox_height, conf = txt_list[j].split(' ')
        cate, x_c, y_c, bbox_width, bbox_height, conf = int(cate), float(x_c), float(y_c), float(bbox_width), float(bbox_height), float(conf)
        x1, y1 = x_c-bbox_width/2, y_c-bbox_height/2 #dim agnostic
        x2, y2 = x_c+bbox_width/2, y_c+bbox_height/2
        mb=list(np.clip([x1,y1,x2,y2], 0, 1)) #make sure we are inside the range [0,1]
        m_boxes.append(mb)
        m_cat.append(cate)
        m_scores.append(conf)
    return m_boxes, m_scores, m_cat

def calculate_consensus_focus(m_folder, skip_box_thr, initial_weights, iou_thr, conf_type):
    """
    :param m_folder: string with the directory that contains the labels of each model. Example: inputs/labels/ has the inferences of three models, so inside the previous folder we find model1/ model2/ model3/
    :param skip_box_thr: list of floats with the confidence gates to judge which bounding box to use. len(confidence_gates) must equal to the number of categories in the label space. 
    :param initial_weights: list of floats with the initial weights to compute the consensus focus. A vector of ones is recommended.
    :param iou_thr: IoU value for boxes to be a match
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    """
    #am_boxes, am_cat, am_scores, am_nb = [], [], [], []
    data_files = sorted([x[0] for x in os.walk(m_folder)]) #folder list
    print(data_files)
    m_folder = data_files[1:][0]+'/'
    data_files1 = [x[2] for x in os.walk(m_folder)][0] #list of txt files, one per image
    consensus_focus_dict = {}
    for i in range(1, len(data_files)):
        consensus_focus_dict[i] = 0 # Esto es para limpiar el diccionario
    source_domain_numbers=len(data_files)-1
    domain_contribution = {frozenset(): 0}
    boxes_contribution = {frozenset(): 0}
    total_comb=sum([math.comb(source_domain_numbers,x) for x in range(source_domain_numbers)])+1 #+1 stands for the empty set
    for i in range(len(data_files1)):
        boxes_list = []
        scores_list = []
        labels_list = []
        for j in range(1,len(data_files)):
            tmp=open(data_files[j]+'/'+data_files1[i]).read().split('\n')
            box_list, score_list, lbl_list = convert_darknet(tmp)
            boxes_list.append(box_list)
            scores_list.append(score_list)
            labels_list.append(lbl_list)
        source_domain_numbers=len(data_files)-1
        for combination_num in range(1, source_domain_numbers + 1):
            combination_list = list(combinations(range(source_domain_numbers), combination_num))
            for combination in combination_list:
                tmp_box=[]
                tmp_scores=[]
                tmp_labels=[]
                tmp_weights=[]
                for element in combination:
                    tmp_box.append(boxes_list[element])
                    tmp_scores.append(scores_list[element])
                    tmp_labels.append(labels_list[element])
                    tmp_weights.append(initial_weights[element]) 
                boxes, scores, labels, nb = weighted_boxes_fusion(tmp_box, tmp_scores, tmp_labels, weights=tmp_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,conf_type=conf_type,allows_overflow=False)
                if len(domain_contribution)!=total_comb:#modificar
                    domain_contribution[frozenset(combination)] = sum(scores*nb) #equation 5.2
                    boxes_contribution[frozenset(combination)] = len(boxes)
                else:
                    domain_contribution[frozenset(combination)] += sum(scores*nb) #equation 5.2
                    boxes_contribution[frozenset(combination)] += len(boxes)
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (
                                                            domain_contribution[frozenset(
                                                                permutation[:permutation.index(source_idx) + 1])]
                                                            - domain_contribution[
                                                                frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict

def ensemble_reweight(cfd,M_i):
    """
    It indicates the relevance of each model according to their consensus focus metrics. Their sum equals 1, and we suggest taking the inverse of each to compute the WBF with such parameters again instead of the initial weights.
    :param cfd: dictionary with the calculated consensus_focus_dict.
    :param M_i: list of ints with the number of images per source dataset. 
    """
    akcf=[]
    suma=sum([M_i[k]*cfd[k+1] for k in range(len(M_i))])
    for c in range(1,len(cfd)+1):
        akcf.append((M_i[c-1]*cfd[c])/suma)
    return akcf

def toDarknet(boxes, labels, scores):
    """
    :param boxes: list of boxes, each with representing a predicted bounding box [x1, y1, x2, y2]. The values go from 0 to 1 and they are image-size agnostic.
    :param labels: list of labels for each predicted bounding box.
    :param scores: list of confidence scores for each predicted bounding box
    """
    darknet=[]
    if len(boxes)==0:
        return None
    for i in range(len(boxes)):
        bbox_width=boxes[i][2]-boxes[i][0]
        bbox_height=boxes[i][3]-boxes[i][1]
        a=(boxes[i][2]+boxes[i][0])/2
        b=(boxes[i][3]+boxes[i][1])/2
        darknet.append([int(labels[i]), a, b, bbox_width, bbox_height, scores[i]])
    return darknet
