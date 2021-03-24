import numpy as np
import time
import sys


def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def filter_segments(segment_predict, videonames, ambilist):
   ind = np.zeros(np.shape(segment_predict)[0])
   for i in range(np.shape(segment_predict)[0]):
      vn = videonames[int(segment_predict[i,0])]
      for a in ambilist:
         if a[0]==vn:
            gt = range(int(round(float(a[2])*25/16)), int(round(float(a[3])*25/16)))
            pd = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
            IoU = float(len(set(gt).intersection(set(pd))))/float(len(set(gt).union(set(pd))))
            if IoU > 0:
               ind[i] = 1
   s = [segment_predict[i,:] for i in range(np.shape(segment_predict)[0]) if ind[i]==0]
   return np.array(s)

# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def getLocMAP(predictions, th, annotation_path, activity_net, valid_id,dataset):

   # gtsegments - temporal segments
   # gtlabels - labels for temporal segments
   # subset - test / validation string indicator for video
   gtsegments = dataset.segments
   # gtsegments = np.load(annotation_path + '/segments.npy')  # 读取视频切分片段信息（groundtruth）
   # print('segments Info:{0} \n {1}'.format(gtsegments.shape,gtsegments[0]))
   # segments Info:(412,) 
   # [[67.5, 75.9], [85.9, 90.6], [139.3, 148.2]]
   gtlabels = dataset.gtlabels
   # gtlabels = np.load(annotation_path + '/labels.npy')
   # print('labels Info:{0}\n{1}'.format(gtlabels.shape,gtlabels[0]))
   # labels Info:(412,)
   # ['Billiards', 'Billiards', 'Billiards']
   # 
   videoname = dataset.videoname
   # videoname = np.load(annotation_path + '/videoname.npy'); videoname = np.array([v.decode('utf-8') for v in videoname])
   subset = dataset.subset; subset = np.array([s.decode('utf-8') for s in subset])
   # subset = np.load(annotation_path + '/subset.npy'); subset = np.array([s.decode('utf-8') for s in subset])
   # print('subset Info:{}\n{}'.format(subset.shape,subset[0]))
   # subset Info:(412,)
   # validation
   classlist = dataset.classlist; classlist = np.array([c.decode('utf-8') for c in classlist])
   # classlist = np.load(annotation_path + '/classlist.npy'); classlist = np.array([c.decode('utf-8') for c in classlist])
   # print('classlist Info:{}\n{}'.format(classlist.shape,classlist[0]))
   # print(classlist)
 #   ['BaseballPitch' 'BasketballDunk' 'Billiards' 'CleanAndJerk' 'CliffDiving'
 # 'CricketBowling' 'CricketShot' 'Diving' 'FrisbeeCatch' 'GolfSwing'
 # 'HammerThrow' 'HighJump' 'JavelinThrow' 'LongJump' 'PoleVault' 'Shotput'
 # 'SoccerPenalty' 'TennisSwing' 'ThrowDiscus' 'VolleyballSpiking']
   # classlist Info:(20,)
   # BaseballPitch
   if not activity_net:
      ambilist = annotation_path + '/Ambiguous_test.txt'
      ambilist = list(open(ambilist,'r'))
      ambilist = [a.strip('\n').split(' ') for a in ambilist]
   else:
      gtsegments = gtsegments[valid_id]
      gtlabels = gtlabels[valid_id]
      videoname = videoname[valid_id]
      subset = subset[valid_id]


   # keep training gtlabels for plotting
   gtltr = []
   train_str = 'validation' if not activity_net else 'training'      # 是否参与训练的标志
   for i,s in enumerate(subset):
      if subset[i]==train_str and len(gtsegments[i]):
         gtltr.append(gtlabels[i])
   gtlabelstr = gtltr                      # 这些是参与训练的视频的labels
      
   # Keep only the test subset annotations
   gts, gtl, vn = [], [], []
   test_str = 'test' if not activity_net else 'validation'           # 是否参与测试的标识
   for i, s in enumerate(subset):
      if subset[i]==test_str:
         gts.append(gtsegments[i])        # 片段信息
         gtl.append(gtlabels[i])          # 标签信息
         vn.append(videoname[i])          # 视频名称
   gtsegments = gts
   gtlabels = gtl
   videoname = vn

   # keep ground truth and predictions for instances with temporal annotations
   gts, gtl, vn, pred = [], [], [], []
   for i, s in enumerate(gtsegments):     # 遍历每个视频的片段信息
      if len(s) > 0:                      # 该片段包含动作片段
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
         pred.append(predictions[i])      # 预测信息
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   predictions = pred

   # which categories have temporal labels ?
   templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))
   # print(templabelcategories)    # 20个动作标签
   # ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
   # 

   # the number index for those categories.
   templabelidx = []
   for t in templabelcategories:
      templabelidx.append(str2ind(t,classlist))
   # print(templabelidx)
   # temp = templabelidx
   if len(predictions[0][0]) == 20:
      # print('YES!')
      templabelidx = [i for i in range(20)]
   # if temp != templabelidx:
      # print('Warning!!!')
      # input()
   # print(templabelidx)
   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
   # YES!
   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
   
   predictions_mod = []
   c_score = []
   for i in range(len(predictions)):
      pr = predictions[i]     # len * 20
      prp = - pr; [prp[:,i].sort() for i in range(np.shape(prp)[1])]; prp=-prp
      end_id = int(np.shape(prp)[0]/8)    # Top-K
      if end_id == 0:
         end_id = 1
      c_s = np.mean(prp[:end_id,:],axis=0)
      # print('c_s Info:{0}\n{1}'.format(c_s.shape,c_s))
      # c_s Info:(20,)
      # [ 1.3365458   1.1334066   0.15256561  0.59985816  0.607889    1.3934332
      #   1.0633978   1.5634567   0.5121401   0.8720228   0.2627355   0.4659454
      #   0.45566705  1.0812104  -0.06136116  0.24608731 -0.0473302   0.07795009
      #   0.5942217   1.0048696 ]
      ind = c_s > 0 if activity_net else (c_s > np.max(c_s)/2)* (c_s > 0)
      # print(ind)
      # [ True  True False False False  True  True  True False  True False False
      #  False  True False False False False False  True]

      # input('please input anything to continue!')
      c_score.append(c_s)
      predictions_mod.append(pr*ind)      # True = 1,False = 0
   predictions = predictions_mod          # 对于那些不在视频中的动作类别列清零


   # For storing per-video detections (with class name, boundaries and confidence for each proposal)
   detection_results = []
   for i,vn in enumerate(videoname):
      detection_results.append([])
      detection_results[i].append(vn)

   ap = []
   gtseg_c = -1
   thr = []
   for c in templabelidx:
      gtseg_c += 1
      segment_predict = []
      # Get list of all predictions for class c
      for i in range(len(predictions)):
         # 对测试集的视频挨个检测
         tmp = predictions[i][:,c]     # 第 c 类
         # threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5  if not activity_net else 0  # 阈值
         threshold = np.mean(tmp)
         thr.append([threshold,np.max(tmp),np.min(tmp)])
         # print(threshold)     # 0.34603163599967957
         vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
         # print('vid_pred Info:{}\n{}'.format(vid_pred.shape,vid_pred))
         # vid_pred Info:(56,)
         # [0. 1. 1. 1. 1. 1. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 0.
         #  1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
         #  1. 0. 0. 0. 1. 0. 0. 0.]
         vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]   # 差分？
         # print('vid_pred_diff Info:{}\n{}'.format(len(vid_pred_diff),vid_pred_diff))
         # vid_pred_diff Info:55
         # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
         # 0.0, -1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
         # start and end of proposals where segments are greater than the average threshold for the class
         s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
         e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
         # print('s:{}\ne:{}'.format(s,e))
         # s:[0, 8, 10, 19, 23, 30, 33, 45, 47, 51]
         # e:[6, 9, 18, 21, 29, 31, 34, 46, 48, 52]
         for j in range(len(s)):
            # Original - Aggregate score is max value of prediction for the class in the proposal and 0.7 * mean(top-k) score of that class for the video
            aggr_score = np.max(tmp[s[j]:e[j]]) + c_score[i][c]
            # append proposal if length is at least 2 segments (16 frames segments @ 25 fps - around 1.25 second)
            if e[j]-s[j]>=2:
               segment_predict.append([i, s[j], e[j], aggr_score])
               detection_results[i].append([classlist[c], s[j], e[j], aggr_score])
      segment_predict = np.array(segment_predict)
      # print('original segment_predict Info:{}\n{}'.format(segment_predict.shape,segment_predict))
      if not activity_net:
         segment_predict = filter_segments(segment_predict, videoname, ambilist)
   
      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         print('No predictions!')
         # print(thr)
         # input()
         return 0
         # continue    # return 0
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]    # 对得分排序
      # print('selected segment_predict Info:{}\n{}'.format(segment_predict.shape,segment_predict))

      # original segment_predict Info:(341, 4)
      # [[  6.           1.          13.           1.22722065]
      #  [  6.          17.          28.           1.27795148]
      #  [  6.          40.          45.           1.35741472]
      #  ...
      #  [191.           9.          15.           1.4395622 ]
      #  [191.          19.          21.           1.32196867]
      #  [191.          23.          25.           1.40317416]]
      # selected segment_predict Info:(336, 4)
      # [[ 62.          28.          46.           2.76686239]
      #  [149.         158.         167.           2.62229276]
      #  [ 66.         171.         175.           2.59842014]
      #  ...
      #  [115.         222.         225.           0.63330257]
      #  [115.         585.         587.           0.63326454]
      #  [115.         250.         252.           0.59105927]]
      # 此处sort前后的矩阵大小是不一样的，因为对于Thumos14存在模棱两可的分类片段，移除了。
      # Create gt list
      # gtsegments,何意？
      # print('gtsegments Info:{}\n{}'.format(len(gtsegments),gtsegments))
      # gtsegments Info:212
      # [[[0.2, 1.1], [11.4, 12.2], [18.6, 20.8], [28.3, 29.7], [1.0, 1.5], [20.8, 22.3], [30.3, 31.7]], [[18.8, 57.3]], [[275.6, 278.8], [281.3, 284.6], [310.2, 313.2], [316.5, 319.3], [347.4, 349.8], [351.5, 354.7], [358.8, 362.5], [390.7, 394.6], [398.9, 402.2], [431.3, 434.5], [436.4, 439.9], [468.4, 472.3]], [[0.4, 12.4], [15.2, 32.0], [40.7, 60.0], [61.8, 77.9]], [[42.7, 43.4], [44.8, 46.9], [48.5, 50.4], [51.5, 52.8], [53.7, 54.3], [56.3, 58.1], [59.9, 61.7], [63.2, 65.3], [66.7, 67.6], [75.7, 76.9], [78.8, 80.0], [81.1, 82.5], [84.5, 85.7], [90.0, 90.9], [94.8, 96.4], [117.7, 119.7], [120.7, 122.1], [123.8, 125.2], [126.5, 128.0], [142.4, 143.8], [145.2, 146.8], [149.0, 150.2], [152.2, 153.4], [155.0, 156.4], [158.6, 159.7], [161.7, 162.5], [173.1, 174.6], [176.6, 177.9], [179.5, 180.6], [182.0, 183.7], [185.7, 186.9], [187.4, 188.2], [190.4, 191.6], [193.3, 194.3], [195.9, 197.0], [198.7, 200.7], [201.4, 203.9], [204.9, 206.1]], [[20.0, 24.0], [29.4, 32.8], [38.2, 40.2], [42.2, 44.4
      segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j],classlist)==c]
      # print('gtsegments[][][0]&[1]{}\n{}'.format(segment_gt[0][1],segment_gt[0][2]))   
      # gtsegments[][][0]&[1]1.4
      # 5.1

      gtpos = len(segment_gt)
      if gtpos == 0:
         print('horrible situation!!')
      # print(gtpos)

      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):
         # 对于第i个预测片段
         flag = 0.
         best_iou = 0
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:
               # 判断是否是对应的视频，是，则进去。
               gt = range(int(round(segment_gt[j][1]*25/16)), int(round(segment_gt[j][2]*25/16)))
               p = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               # remove gt segment if IoU is greater than threshold (since predicted segments are sorted according to their 'actioness' scores)
               if IoU >= th:
                  flag = 1.
                  if IoU > best_iou:
                     best_iou = IoU
                     best_j = j        # 因为一个预测，只能对应一个真实的segment，此处即标记最佳的片段，后期，删除之。
         if flag > 0:
            del segment_gt[best_j]
         tp.append(flag)
         fp.append(1.-flag)
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      if sum(tp)==0:
         # 说明预测的segment不符合要求的IoU
         print('Blind of your model!')
         prc = 0.
      else:
         cur_prec = tp_c / (fp_c+tp_c)
         cur_rec = 1. * tp_c / gtpos
         prc = _ap_from_pr(cur_prec, cur_rec)
      ap.append(prc)

   return 100*np.mean(ap)
  

def getDetectionMAP(predictions, annotation_path, activity_net=False,dataset=None ,valid_id=None):
   # print(predictions[0].shape)    # (52, 20)
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
   if activity_net:
      iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
   dmap_list = []

   for iou in iou_list:
      print('Testing for IoU %f' %iou)
      dmap_list.append(getLocMAP(predictions, iou, annotation_path, activity_net, valid_id,dataset))

   return dmap_list, iou_list

