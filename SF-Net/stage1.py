import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time
import random

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import math


def CLSLOSS(logits, seq_len, batch_size, labels, device):
    lab = F.softmax(labels, dim=1)
    clsloss = -torch.mean(torch.sum(Variable(lab) * F.log_softmax(logits, dim=1), dim=1), dim=0)
    return clsloss


def InPOINTSLOSS(logits, batch_size, point_idx, gtlabels, device, itr, len):
    lab = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        labels = torch.from_numpy(np.array([gtlabels[i]])).float().to(device)
        for k, pt in enumerate(point_idx[i]):
            length = random.randint(0, len)
            start, end = pt - length, pt + length + 1
            if start < 0:
                start = 0
            if end > logits[i].shape[0]:
                end = logits[i].shape[0]
            tmp_logits = torch.zeros(0).to(device)
            for se in range(start, end):
                tmp_logits = torch.cat([tmp_logits, logits[i][[se]]], dim=0)
            instance_logits = torch.cat([instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)
            lab = torch.cat([lab, labels[0][[k]]], dim=0)

    Inploss = -torch.mean(torch.sum(Variable(lab) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)

    return Inploss



def ACTIONLOSS(action, seq_len, batch_size, point_idx, device, len):
    lab = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    target = torch.ones([1, 1], dtype=torch.float).to(device)
    for i in range(batch_size):
        flag = False
        tmp_logits = torch.zeros(0).to(device)
        for idx in point_idx[i]:
            flag = True
            length = random.randint(0, len)
            start, end = idx - length, idx + length + 1
            if start < 0:
                start = 0
            if end > action[i].shape[0]:
                end = action[i].shape[0]
            for se in range(start, end):
                tmp_logits = torch.cat([tmp_logits, action[i][[se]]], dim=0)
        if flag:
            lab = torch.cat([lab, target], dim=0)
            instance_logits = torch.cat([instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)

    actloss = F.binary_cross_entropy_with_logits(instance_logits, lab)
    return actloss




def EmbeddingLoss(embfeature, batch_size, gtlabels, fpAct, fpBg, FLAG=True):
    total_loss = 0.0
    actloss, actbgloss = [], []
    for i in range(batch_size):
        # input()
        label, feat, frame2Act = gtlabels[i], embfeature[i], fpAct[i]
        # print(frame2Act)
        if FLAG:
            frame2BG = fpBg[i]
        else:
            frame2BG = []
        classDict = {}  # ç±»åˆ«1ï¼špoint1ï¼Œpoint2ï¼Œã€‚ã€‚ã€‚ã€?
        for k, lab in enumerate(label):
            c = int(np.where(lab == 1)[0])
            if c in classDict.keys():
                classDict[c].append(k)
            else:
                classDict[c] = [k]
        actionList = []  # each class action instance
        allAction = []  # all class action instance
        actbgList = frame2BG  # background
        for cd, value in classDict.items():
            tmp = []
            for it in value:
                allAction.extend(frame2Act[it])
                tmp.extend(frame2Act[it])
            actionList.append(tmp)
        F1, F2, B1, B2 = [], [], [], []
        # è®¡ç®—actionloss and actbgloss
        if actionList:
            flag = False
            for act in actionList:
                tmpList = []
                for tmp in act:
                    tmpList.append(tmp)
                random.shuffle(tmpList)
                if len(tmpList) > 1:
                    flag = True
                    F1.extend(tmpList[0:len(tmpList) // 2])
                    F2.extend(tmpList[len(tmpList) // 2:(len(tmpList) // 2) * 2])
            if flag:
                actloss.append(np.mean([sim(feat[[idx1]], feat[[idx2]])[0] for idx1, idx2 in zip(F1, F2)]))
            length = min(len(allAction), len(actbgList))
            if length > 0:
                random.shuffle(allAction)
                random.shuffle(actbgList)

                B1 = allAction[0:length]
                B2 = actbgList[0:length]
                actbgloss.append(-np.mean(
                    [np.log(1 - Reverse(sim(feat[[idx1]], feat[[idx2]])[0])) for idx1, idx2 in zip(B1, B2) if
                     Reverse(sim(feat[[idx1]], feat[[idx2]])[0]) != 1]))

    if actloss:
        total_loss += np.mean(actloss)
    if actbgloss:
        total_loss += np.mean(actbgloss)
    return total_loss



def train(itr, dataset, args, model, optimizer, logger, device):
    # Batch fprop
    features, labels, gtlabel, count_labels, point_indexs = dataset.load_data()
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    feature_f = features[:, :, 1024:]
    feature_r = features[:, :, :1024]
    labels = torch.from_numpy(labels).float().to(device)
    count_labels = torch.from_numpy(count_labels).float().to(device)

    logits_f, logits_r, tcam, score_f, score_r, score_all, embfeat_f, embfeat_r, embfeat, action_f, action_r, action_all = model(
        Variable(features), device, seq_len=torch.from_numpy(seq_len).to(device))

    total_loss = 0.0

    actionExit_f = LIST(point_indexs)
    actionExit_r = LIST(point_indexs)
    actionExit = LIST(point_indexs)
    iploss_f = InPOINTSLOSS(logits_f, args.batch_size, point_indexs, gtlabel, device, itr, len=2)
    iploss_r = InPOINTSLOSS(logits_r, args.batch_size, point_indexs, gtlabel, device, itr, len=2)
    iploss_final = InPOINTSLOSS(tcam, args.batch_size, point_indexs, gtlabel, device, itr, len=2)
    iploss = iploss_f + iploss_r + iploss_final

    actloss_f = ACTIONLOSS(action_f, seq_len, args.batch_size, point_indexs, device, len=2)
    actloss_r = ACTIONLOSS(action_r, seq_len, args.batch_size, point_indexs, device, len=2)
    actloss_final = ACTIONLOSS(action_all, seq_len, args.batch_size, point_indexs, device, len=2)
    actloss = actloss_r + actloss_f + actloss_final

    total_loss += iploss + actloss

    bgMaybe_f, bgMaybe_r, bgMaybe = [], [], []

    embloss_f = EmbeddingLoss(embfeat_f, args.batch_size, gtlabel, actionExit_f, bgMaybe_f, FLAG)
    embloss_r = EmbeddingLoss(embfeat_r, args.batch_size, gtlabel, actionExit_r, bgMaybe_r, FLAG)
    embloss_final = EmbeddingLoss(embfeat, args.batch_size, gtlabel, actionExit, bgMaybe, FLAG)
    embloss = embloss_f + embloss_r + embloss_final
    total_loss += embloss

    clsloss_f = CLSLOSS(score_f, seq_len, args.batch_size, labels, device)
    clsloss_r = CLSLOSS(score_r, seq_len, args.batch_size, labels, device)
    clsloss_final = CLSLOSS(score_all, seq_len, args.batch_size, labels, device)
    clsloss = clsloss_f + clsloss_r + clsloss_final
    total_loss += clsloss

    logger.log_value('total_loss', total_loss, itr)
    print('Iteration: %d, Loss: %.3f' % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    if total_loss > 0:
        total_loss.backward()
    if total_loss > 0:
        optimizer.step()