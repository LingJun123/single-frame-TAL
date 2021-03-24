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




def BackGround(logits, action, seq_len, batch_size, point_idx, device):
    k = np.ceil(seq_len / 8).astype('int32')
    Idx2BG = []
    for i in range(batch_size):
        tmplogits = logits[i][:, -1][:seq_len[i]].data.cpu().numpy().reshape(
            logits[i][:, -1][:seq_len[i]].data.cpu().numpy().shape[0], )
        tmpaction = -action[i][:seq_len[i]].data.cpu().numpy().reshape(
            action[i][:seq_len[i]].data.cpu().numpy().shape[0], )
        tmp = tmplogits + tmpaction
        gtIndex = point_idx[i]
        _, index = torch.topk(torch.from_numpy(tmp), k=int(k[i]), dim=0, largest=True)
        Idx2BG.append([idx.item() for idx in index if idx not in gtIndex])
    return Idx2BG


def BACKGROUNDLOSS(logits, action, seq_len, batch_size, point_idx, device):
    neg_lab = torch.zeros([1, 21], dtype=torch.float).to(device)
    neg_lab[0, 20] = 1
    weak_lab = torch.zeros(0).to(device)
    target = torch.ones([1, 1], dtype=torch.float).to(device)
    neg_label = torch.zeros(0).to(device)
    neg_instance_logits = torch.zeros(0).to(device)
    neg_action_instance_logits = torch.zeros(0).to(device)
    k = np.ceil(seq_len / 8).astype('int32')
    for i in range(batch_size):
        index = point_idx[i]
        tmp_logits = torch.zeros(0).to(device)
        tmp_action_logits = torch.zeros(0).to(device)
        FLAG = False
        for idx in index:
            FLAG = True
            tmp_logits = torch.cat([tmp_logits, logits[i][[idx]].reshape(1, -1)], dim=0)
            tmp_action_logits = torch.cat([tmp_action_logits, action[i][[idx]].reshape(1, -1)], dim=0)
        if FLAG:
            neg_label = torch.cat([neg_label, neg_lab], dim=0)
            neg_instance_logits = torch.cat([neg_instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)

            weak_lab = torch.cat([weak_lab, target], dim=0)
            neg_action_instance_logits = torch.cat(
                [neg_action_instance_logits, torch.mean(tmp_action_logits, 0, keepdim=True)], dim=0)

    neg_SegLoss = -0.1 * torch.mean(torch.sum(Variable(neg_label) * F.log_softmax(neg_instance_logits, dim=1), dim=1),
                                    dim=0)
    neg_ActLoss = 0.1 * F.binary_cross_entropy(1 - torch.sigmoid(neg_action_instance_logits), weak_lab)
    return neg_SegLoss + neg_ActLoss


def sim(f1, f2):
    f1 = torch.transpose(f1, 1, 0)
    f2 = torch.transpose(f2, 1, 0)
    sim_loss = 1 - torch.sum(f1 * f2, dim=0) / (torch.norm(f1, 2, dim=0) * torch.norm(f2, 2, dim=0))
    return np.around(sim_loss.data.cpu().numpy(), decimals=4)


def getS1(elem):
    return elem[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Reverse(x):
    return (np.exp(-x))


def spliteCAS(CAS, interval, point_idx, actionNoExitIndex=[]):
    if len(point_idx) > 1:
        for j in range(1, len(point_idx)):
            # print(interval)
            Exist = [(t, ex) for t, ex in enumerate(interval) if (point_idx[j - 1] in ex and point_idx[j] in ex)]
            if Exist:
                idxtmp = Exist[0][0];
                Exist = Exist[0][1]
                if point_idx[j] - point_idx[j - 1] > 1:
                    minVal = min(CAS[point_idx[j - 1] + 1:point_idx[j]])
                    minIdx = [t + point_idx[j - 1] + 1 for t, m in enumerate(CAS[point_idx[j - 1] + 1:point_idx[j]]) if
                              m == minVal]
                    actionNoExitIndex.append([idx for idx in minIdx])

                    interval[idxtmp] = []
                    AD = 0
                    if Exist[0] - AD > 0:
                        interval.append([v for v in range(Exist[0] - AD, minIdx[0])])
                    else:
                        interval.append([v for v in range(Exist[0], minIdx[0])])
                    # print([v for v in range(Exist[0],minIdx)])
                    interval.append([v for v in range(minIdx[-1] + 1, Exist[-1] + 1)])
    interval = [Int for Int in interval if Int]
    return interval, actionNoExitIndex


def tmpSpliteCAS(CAS, pt, point_idx, interval):
    idx = [dx for dx, p in enumerate(point_idx) if p == pt][0]
    left = idx - 1
    right = idx + 1
    lp = interval[0]
    rp = interval[-1]
    if left > -1:
        if point_idx[left] in interval:
            if pt - point_idx[left] > 1:  # 3 - 1 = 2
                minVal = min(CAS[point_idx[left] + 1:point_idx[idx]])
                minIdx = \
                [t + point_idx[left] + 1 for t, m in enumerate(CAS[point_idx[left] + 1:point_idx[idx]]) if m == minVal][
                    0]
                lp = minIdx
            else:
                lp = pt

    if right < len(point_idx):
        if point_idx[right] in interval:
            if point_idx[right] - pt > 1:  # 3 - 1 = 2
                minVal = min(CAS[point_idx[idx] + 1:point_idx[right]])
                minIdx = [t + pt + 1 for t, m in enumerate(CAS[point_idx[idx] + 1:point_idx[right]]) if m == minVal][0]
                rp = minIdx
            else:
                rp = pt
    return [m for m in range(lp, rp + 1)]


def deleteWeakSegment(feature, CAS, Action, interval, PotITL, point_idx, rate=1.0):
    # cas = getNewCAS(CAS,Action)
    cas = CAS + Action
    for pt in point_idx:
        interval.sort()
        f1 = feature[[pt]]
        ITL = [(idx, itl) for idx, itl in enumerate(interval) if pt in itl and len(itl) > 1]
        if ITL:
            origin = ITL[0][1]
            PotTimeITL = [itl for itl in PotITL if pt in itl][0]
            tmpITL = getHighConfidenceSegment(cas, origin, pt)
            if len(tmpITL) > 1:
                ITL = tmpITL
            else:
                ITL = origin

            score = []
            for idx in ITL:
                f2 = feature[[idx]]
                tmpSim = sim(f1, f2)[0]
                if idx != pt:
                    score.append(cas[idx] * Reverse(tmpSim))
            # thre = np.mean(score)  # ?junzhi1?zhongzhi1?
            thre = np.min(score) + (np.max(score) - np.min(score)) * 0.5
            # threshold = 0.0
            # SeedTemporalGrowing: STG
            idxLeft, idxRight = pt, pt
            for k in range(len(PotTimeITL)):
                f2 = feature[[PotTimeITL[k]]]
                tmpSim = sim(f1, f2)[0]
                aggrate = cas[PotTimeITL[k]] * Reverse(tmpSim)
                if PotTimeITL[k] < idxLeft and aggrate >= thre:
                    idxLeft = PotTimeITL[k]
                    break
            for k in range(len(PotTimeITL) - 1, -1, -1):
                f2 = feature[[PotTimeITL[k]]]
                tmpSim = sim(f1, f2)[0]
                aggrate = cas[PotTimeITL[k]] * Reverse(tmpSim)
                if PotTimeITL[k] > idxRight and aggrate >= thre:
                    idxRight = PotTimeITL[k]
                    break

            proposalITL = [d for d in range(idxLeft, idxRight + 1)]
            delITL = []
            for it in interval:
                if set(it).intersection(set(proposalITL)):
                    delITL.append(it)
            for it in delITL:
                interval.remove(it)
            interval.append(proposalITL)

    return interval


def getHighConfidenceSegment(logits, predSegment, seed, minRate=0.5):
    '''
    1.the ball stay in the smallest prediction of the snipppet.
    2.the ball simulate the true world,and the low prodiction maybe not the smallese prediction.
    '''
    tmp = logits[predSegment[0]:predSegment[-1] + 1]
    difference = max(tmp) - min(tmp)
    minThreshold = min(tmp) + minRate * difference
    idx = seed - predSegment[0]
    if minThreshold > tmp[idx]:
        minThreshold = tmp[idx]
    # print(minThreshold)
    # difThreshold = difRate * difference
    idxLeft, idxRight = idx, idx
    # core zone
    # left
    for k in range(idx - 1, -1, -1):
        if tmp[k] >= minThreshold:
            idxLeft = k
        else:
            break
    # right
    for k in range(idx + 1, len(tmp)):
        if tmp[k] >= minThreshold:
            idxRight = k
        else:
            break
    refineITL = [k + predSegment[0] for k in range(idxLeft, idxRight + 1)]
    return refineITL


def RemoveEdgeSnippet(segment, score, threshold):
    tmp = score[segment[0]:segment[-1] + 1]
    add = segment[0]
    for k in range(len(segment)):
        if tmp[k] >= threshold:
            idxLeft = k
            break
    for k in range(len(segment) - 1, -1, -1):
        if tmp[k] >= threshold:
            idxRight = k
            break
    refineITL = [k + add for k in range(idxLeft, idxRight + 1)]
    return refineITL


def getPotTiITL(itl, actitl, ptSet, length):
    PotITL = []
    CNT_1 = CNT_2 = 0
    for pt in ptSet:
        tmpPotITL = [it for it in itl if pt in it]
        actPotITL = [it for it in actitl if pt in it]
        if tmpPotITL and actPotITL:
            # MaxITL = [k for k in range(tmpPotITL[0][0]-2 * len(tmpPotITL[0]),tmpPotITL[0][-1]+ 2 * len(tmpPotITL[0])) if k >-1 and k < length]
            ITL = list(set(tmpPotITL[0]).union(set(actPotITL[0])))
            # ITL = list(set(MaxITL).intersection(set(tmpITL)))
            ITL.sort()
            PotITL.append(ITL)
        elif tmpPotITL:
            PotITL.append(tmpPotITL[0])
    return PotITL


def FineIndex(pointIndex, actionIndex):
    NewIndex = []
    for pt in pointIndex:
        NewIndex.append([Idx for Idx in actionIndex if pt in Idx][0])
    return NewIndex


def SEGMENT(feature, logits, action, seq_len, batch_size, point_idx, gtlabels, args, device):
    '''
    gtlabel:list[array(),....]
    '''
    actionExit = []
    actionNoExit = []
    for i in range(batch_size):
        GT = {}
        actionExitIndex = []
        actionNoExitIndex = []
        for idx, gtL in zip(point_idx[i], gtlabels[i]):
            # print(idx,gtL)      # 276 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            c = int(np.where(gtL == 1)[0])
            if c in GT.keys():
                temp = GT[c]
                temp.append(idx)
                GT[c] = temp
            else:
                GT[c] = [idx]

        for c in GT.keys():  # åŠ¨ä½œå­˜åœ¨
            tmp = logits[i][:, c][:seq_len[i]].data.cpu().numpy().reshape(
                logits[i][:, c][:seq_len[i]].data.cpu().numpy().shape[0], )
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp)) * 0.5 if not args.activity_net else 0  # é˜?
            vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]

            interval = [[k for k in range(si, ei)] for si, ei in zip(s, e)]  # æ¯ ä¸ªsegmentçš„åŒº?siä¸ åŒ…?
            # [0,1,1,0]->[1,0,-1]->(0,2)->[[0,1],]        # ä¸ è¦ æƒ³å¤ªå¤šï¼Œåœ¨å‰ é ¢å·²ç» æŠŠé¦–å°¾è¡¥é›¶äº?

            GT[c] = list(set(GT[c]))  # å ‡åº
            GT[c].sort()

            # å¤„ç †åŒºæ®µï¼ŒæŠŠå­˜åœ¨å¤šç‚¹çš„è¿›è¡Œæ‹†?
            interval, actionNoExitIndex = spliteCAS(tmp, interval, GT[c], actionNoExitIndex)

            act = action[i][:seq_len[i]].data.cpu().numpy().reshape(
                action[i][:seq_len[i]].data.cpu().numpy().shape[0], )
            threshold = np.max(act) - (np.max(act) - np.min(act)) * 0.5 if not args.activity_net else 0  # é˜?
            vid_pred = np.concatenate([np.zeros(1), (act > threshold).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]

            actITL = [[k for k in range(si, ei)] for si, ei in zip(s, e)]  # æ¯ ä¸ªsegmentçš„åŒº?siä¸ åŒ…?
            actITL, _ = spliteCAS(act, actITL, GT[c])

            PotITL = getPotTiITL(interval, actITL, GT[c], len(act))

            interval = deleteWeakSegment(feature[i], tmp, act, interval, PotITL, GT[c], rate=1)
            # æž„é€ æ­£ä¾‹ï¼Œå ³å­˜åœ¨pointçš„åŒº?
            index_exit = []  # ä¿ å­˜å­˜åœ¨pointçš„åŒºæ®µçš„index
            for idx in GT[c]:
                ind = int([i for i, p in enumerate(point_idx[i]) if p == idx][0])
                ITL = [itl for itl in interval if idx in itl]  # ITL æ˜¯idxå­˜åœ¨çš„åŒºæ®µï¼Œæœ‰å ¯èƒ½ä¸ºNULL
                if ITL:
                    posSegment = ITL[0]
                    actionExitIndex.append([tmpIdx for tmpIdx in posSegment])
                    interval.remove(ITL[0])

                else:
                    length = 0
                    start, end = idx - length, idx + length + 1
                    if start < 0:
                        start = 0
                    if end > logits[i].shape[0]:
                        end = logits[i].shape[0]
                    actionExitIndex.append([dx for dx in range(start, end)])

            for ITL in interval:
                actionNoExitIndex.append([idx for idx in ITL])
        actionExitIndex = FineIndex(point_idx[i], actionExitIndex)
        actionExit.append(actionExitIndex)
        actionNoExit.append(actionNoExitIndex)
    return (actionExit, actionNoExit)


def SEGMENTLOSS(logits, batch_size, point_idx, gtlabels, actionExit, actionNoExit, device):
    lab = torch.zeros(0).to(device)
    neg_lab = torch.zeros([1, 21], dtype=torch.float).to(device)
    neg_lab[0, 20] = 1
    weak_neg_label = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    weak_neg_instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        # ç¡®å®šæ¯ ä¸€ä¸ªè§†é¢‘ä¸­å‡ºçŽ°çš„åŠ¨ä½œç±»?
        GT = {}
        labels = torch.from_numpy(np.array([gtlabels[i]])).float().to(device)
        for idx, gtL in zip(point_idx[i], gtlabels[i]):
            # print(idx,gtL)      # 276 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            c = int(np.where(gtL == 1)[0])
            if c in GT.keys():
                temp = GT[c]
                temp.append(idx)
                GT[c] = temp
            else:
                GT[c] = [idx]

        # ä¸Šè¿°è¿‡ç¨‹ï¼Œæ±‚å‡ºäº†ç±»åˆ«cå¯¹åº”çš„å ¯èƒ½åŒº?
        for c in GT.keys():  # åŠ¨ä½œå­˜åœ¨
            # æž„é€ æ­£ä¾‹ï¼Œå ³å­˜åœ¨pointçš„åŒº?
            for idx in GT[c]:
                FLAG = False
                # print('idx:{}'.format(idx))
                tmp_logits = torch.zeros(0).to(device)
                ITL = [itl for itl in actionExit[i] if idx in itl][0]  # ITL æ˜¯idxå­˜åœ¨çš„åŒº?
                for Idx in ITL:
                    FLAG = True
                    tmp_logits = torch.cat([tmp_logits, logits[i][[Idx]]], dim=0)
                if FLAG:
                    ind = int([i for i, p in enumerate(point_idx[i]) if p == idx][0])
                    lab = torch.cat([lab, labels[0][[ind]]], dim=0)
                    instance_logits = torch.cat([instance_logits, torch.mean(tmp_logits, 0, keepdim=True)],
                                                dim=0)  # å –å¹³?

        # æž„é€ è´Ÿä¾‹ï¼Œå ³å¾—åˆ†å¾ˆé«˜ï¼Œä½†æ˜¯ä¸ å­˜åœ¨pointçš„åŒº?
        for ITL in actionNoExit[i]:
            tmp_logits = torch.zeros(0).to(device)
            FLAG = False
            for Idx in ITL:
                FLAG = True
                tmp_logits = torch.cat([tmp_logits, logits[i][[Idx]]], dim=0)
            if FLAG:
                weak_neg_label = torch.cat([weak_neg_label, neg_lab], dim=0)
                weak_neg_instance_logits = torch.cat(
                    [weak_neg_instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)  # å –å¹³?
    weak_neg_SegLoss = -torch.mean(
        torch.sum(Variable(weak_neg_label) * F.log_softmax(weak_neg_instance_logits, dim=1), dim=1), dim=0)
    SegLoss = -torch.mean(torch.sum(Variable(lab) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    # penalize = torch.mean(torch.sum(F.softmax(instance_logits, dim=1), dim=1), dim=0)
    # SegLoss += 0.001 * penalize
    return SegLoss + 0.1 * weak_neg_SegLoss


def UnionActionNoExit(ANE_F, ANE_R, batch_size):
    actionNoExit = []
    for i in range(batch_size):
        TMP = []
        for ane_f in ANE_F[i]:
            # print(ane_f)
            # input()
            t_max = ane_f[-1]
            t_min = ane_f[0]
            # FLAG = True
            for ane_r in ANE_R[i]:
                if list(set(ane_f).intersection(set(ane_r))):  # äº¤é›†ä¸ ä¸º?
                    # FLAG = False
                    ANE_R[i].remove(ane_r)
                    if t_min > ane_r[0]:
                        t_min = ane_r[0]
                    if t_max < ane_r[-1]:
                        t_max = ane_r[-1]
            TMP.append([i for i in range(t_min, t_max + 1)])
        for ane_r in ANE_R:
            TMP.append(ane_r)
        actionNoExit.append(TMP)
    return actionNoExit


def InterActionExit(ANE_F, ANE_R, point_indexs, batch_size):
    actionExit = []
    for i in range(batch_size):
        TMP = []
        for idx in point_indexs[i]:
            ITL_f = [itl for itl in ANE_F[i] if idx in itl][0]
            ITL_r = [itl for itl in ANE_R[i] if idx in itl][0]
            # TMP.append(list(set(ITL_f).union(set(ITL_r))))
            TMP.append(list(set(ITL_f).intersection(set(ITL_r))))
        actionExit.append(TMP)
    return actionExit


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


def ACTION2SEGLOSS(action, actionExit, actionNoExit, batch_size, device):
    lab = torch.zeros(0).to(device)
    weak_lab = torch.zeros(0).to(device)
    instance_logits = torch.zeros(0).to(device)
    neg_instance_logits = torch.zeros(0).to(device)
    weak_neg_instance_logits = torch.zeros(0).to(device)
    target = torch.ones([1, 1], dtype=torch.float).to(device)
    for i in range(batch_size):
        tmp_logits = torch.zeros(0).to(device)
        FLAG = False
        for itl in actionExit[i]:
            for idx in itl:
                FLAG = True
                tmp_logits = torch.cat([tmp_logits, action[i][[idx]]], dim=0)
        if FLAG:
            lab = torch.cat([lab, target], dim=0)
            instance_logits = torch.cat([instance_logits, torch.mean(tmp_logits, 0, keepdim=True)], dim=0)
        # å¤„ç †å¼±è´Ÿ?
        tmp_logits = torch.zeros(0).to(device)
        FLAG = False
        for itl in actionNoExit[i]:
            for idx in itl:
                FLAG = True
                tmp_logits = torch.cat([tmp_logits, action[i][[idx]]], dim=0)
        if FLAG:
            weak_lab = torch.cat([weak_lab, target], dim=0)
            weak_neg_instance_logits = torch.cat([weak_neg_instance_logits, torch.mean(tmp_logits, 0, keepdim=True)],
                                                 dim=0)

    actloss = F.binary_cross_entropy_with_logits(instance_logits, lab)
    # actloss +=  0.05 * F.binary_cross_entropy(1-torch.sigmoid(neg_instance_logits),lab)
    # if weak_flag:
    actloss += 0.1 * F.binary_cross_entropy(1 - torch.sigmoid(weak_neg_instance_logits), weak_lab)
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


def LIST(actExit):
    Exit = []
    for i in range(len(actExit)):
        tmp = []
        for it in actExit[i]:
            if not isinstance(it, list):
                it = [it]
            tmp.append(it)
        Exit.append(tmp)
    return Exit


def UNION(actExit):
    Exit = []
    for i in range(len(actExit)):
        tmp = []
        for it in actExit[i]:
            tmp.extend(it)
        Exit.append(tmp)
    return Exit


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
    FLAG = False
    if itr % 2:
        FLAG = True
        actionExit_f, actionNoExit_f = SEGMENT(feature_f, logits_f, action_f, seq_len, args.batch_size,
                                               point_indexs, gtlabel, args, device)
        actionExit_r, actionNoExit_r = SEGMENT(feature_r, logits_r, action_r, seq_len, args.batch_size,
                                               point_indexs, gtlabel, args, device)
        actionExit, actionNoExit = SEGMENT(features, tcam, action_all, seq_len, args.batch_size, point_indexs,
                                           gtlabel, args, device)

        segloss_f = SEGMENTLOSS(logits_f, args.batch_size, point_indexs, gtlabel, actionExit_f, actionNoExit_f, device)
        segloss_r = SEGMENTLOSS(logits_r, args.batch_size, point_indexs, gtlabel, actionExit_r, actionNoExit_r, device)
        segment_final = SEGMENTLOSS(tcam, args.batch_size, point_indexs, gtlabel, actionExit, actionNoExit, device)

        segloss = segloss_f + segloss_r + segment_final

        actloss_f = ACTION2SEGLOSS(action_f, actionExit_f, actionNoExit_f, args.batch_size, device)
        actloss_r = ACTION2SEGLOSS(action_r, actionExit_r, actionNoExit_r, args.batch_size, device)
        actloss_final = ACTION2SEGLOSS(action_all, actionExit, actionNoExit, args.batch_size, device)
        actloss = actloss_r + actloss_f + actloss_final
        total_loss += segloss + actloss
    else:
        iploss_f = InPOINTSLOSS(logits_f, args.batch_size, point_indexs, gtlabel, device, itr, len=0)
        iploss_r = InPOINTSLOSS(logits_r, args.batch_size, point_indexs, gtlabel, device, itr, len=0)
        iploss_final = InPOINTSLOSS(tcam, args.batch_size, point_indexs, gtlabel, device, itr, len=0)
        iploss = iploss_f + iploss_r + iploss_final

        actloss_f = ACTIONLOSS(action_f, seq_len, args.batch_size, point_indexs, device, len=0)
        actloss_r = ACTIONLOSS(action_r, seq_len, args.batch_size, point_indexs, device, len=0)
        actloss_final = ACTIONLOSS(action_all, seq_len, args.batch_size, point_indexs, device, len=0)
        actloss = actloss_r + actloss_f + actloss_final

        total_loss += iploss + actloss

    bgMaybe_f, bgMaybe_r, bgMaybe = [], [], []
    if FLAG:
        bgMaybe_f = BackGround(logits_f, action_f, seq_len, args.batch_size, UNION(actionExit_f), device)
        bgMaybe_r = BackGround(logits_r, action_r, seq_len, args.batch_size, UNION(actionExit_r), device)
        bgMaybe = BackGround(tcam, action_all, seq_len, args.batch_size, UNION(actionExit), device)

    embloss_f = EmbeddingLoss(embfeat_f, args.batch_size, gtlabel, actionExit_f, bgMaybe_f, FLAG)
    embloss_r = EmbeddingLoss(embfeat_r, args.batch_size, gtlabel, actionExit_r, bgMaybe_r, FLAG)
    embloss_final = EmbeddingLoss(embfeat, args.batch_size, gtlabel, actionExit, bgMaybe, FLAG)
    embloss = embloss_f + embloss_r + embloss_final
    total_loss += embloss

    if FLAG:
        bgloss_f = BACKGROUNDLOSS(logits_f, action_f, seq_len, args.batch_size, bgMaybe_f, device)
        bgloss_r = BACKGROUNDLOSS(logits_r, action_r, seq_len, args.batch_size, bgMaybe_r, device)
        bgloss_final = BACKGROUNDLOSS(tcam, action_all, seq_len, args.batch_size, bgMaybe, device)
        bgloss = bgloss_f + bgloss_r + bgloss_final
        total_loss += bgloss

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