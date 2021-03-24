import numpy as np
import glob
import utils
import time
import torch
import os

# SLURM_JOBID = os.getenv('SLURM_JOBID')
# SLURM_JOB_USER = os.getenv('SLURM_JOB_USER')
# SLURM_JOB_ID=os.getenv('SLURM_JOB_ID')
dataset_dir = '/mnt/10601003/dataset/Thumos14reduced-Annotations/'

class Dataset():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.length = args.length
        # self.path_to_features = '/mnt/ssd/wb/pointData/'+self.dataset_name + '-I3D-JOINTFeatures.npy'
        self.path_to_annotations = dataset_dir
        self.features = np.load(self.path_to_annotations + self.dataset_name + '-I3D-JOINTFeatures.npy', encoding='bytes',allow_pickle=True)
        self.segments = np.load(self.path_to_annotations + 'segments.npy',allow_pickle=True)
        self.gtlabels = np.load(self.path_to_annotations + 'labels.npy',allow_pickle=True)
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy',allow_pickle=True)     # Specific to Thumos14
        # print(len(self.gtlabels))
        # print(len(self.gtlabels))
        # print(self.gtlabels)
        # if self.gtlabels.all() == self.labels.all():
        #     print('111111111111')
        self.activity_net = args.activity_net
        # self.classlist20 = None
        # if not self.activity_net:
        #     self.classlist20 = np.load(self.path_to_annotations + '/classlist_20classes.npy')   
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy',allow_pickle=True)
        self.subset = np.load(self.path_to_annotations + 'subset.npy',allow_pickle=True)
        self.duration = np.load(self.path_to_annotations + 'duration.npy',allow_pickle=True)
        self.videoname = np.load(self.path_to_annotations + 'videoname.npy',allow_pickle=True)
        self.lst_valid = None
        if self.activity_net:
            lst_valid = []
            for i in range(self.features.shape[0]):
                feat = self.features[i]
                mxlen = np.sum(np.max(np.abs(feat), axis=1) > 0, axis=0)
                # Remove videos with less than 5 segments
                if mxlen > 5:
                    lst_valid.append(i)
            self.lst_valid = lst_valid
            if len(lst_valid) != self.features.shape[0]:
                self.features = self.features[lst_valid]
                self.subset = self.subset[lst_valid]
                self.videoname = self.videoname[lst_valid]
                self.duration = self.duration[lst_valid]
                self.gtlabels = self.gtlabels[lst_valid]
                self.labels = self.labels[lst_valid]
                self.segments = self.segments[lst_valid]

        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.currentvalidx = 0
        # print(self.classlist.shape,self.labels.shape)     # (20,)
        # print(self.classlist,self.labels)
        # print()
        self.labels_multihot = [utils.strlist2multihot(labs,self.classlist) for labs in self.labels]
        # print(self.labels_multihot[0])
        self.train_test_idx()
        self.classwise_feature_mapping()
        # self.labels101to20 = None
        # self.labels101to20 = None if self.activity_net else np.array(self.classes101to20())
        self.class_order = self.get_class_id()
        self.count_labels = self.get_count()
        # print()

        # For Point!
        self.point = np.load(self.path_to_annotations + 'point.npy',allow_pickle=True)       # point
        self.points2Idx = self.get_point()         							# 落在第一个训练的片段�?
        self.labels_onehot = []
        self.classlist = np.append(self.classlist,'BackGround'.encode('utf-8'))
        for labels in self.gtlabels:
        	# print(labels)
        	# onehot = utils.strlist2multihot([labels[0]],self.classlist)
        	# print(onehot)
        	# input()
        	self.labels_onehot.append([utils.strlist2multihot([labs],self.classlist) for labs in labels])
        	# print(self.labels_onehot)
        	# input()
  #       print("self.labels_onehot Info:{}\n{}".format(len(self.labels_onehot),self.labels_onehot[0]))
  #       self.labels_onehot Info:412
		# [array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
		#        0., 0., 0.]), array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
		#        0., 0., 0.]), array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
		#        0., 0., 0.])]
        # self.labels_onehot = [utils.strlist2multihot(labs,self.classlist) for labels in self.labels for labs in labels]
        # self.points2Idx = self.points2Idx


    def get_point(self):
        # point  of instances of each category present in the video
        points = []
        for i in range(len(self.point)):

            tmp = [int(pt*25/16) for pt in self.point[i]  if int(round((pt*25/16))) < self.features[i].shape[0]]
            
            points.append(tmp)
        points = np.array(points)
        return points


    def train_test_idx(self):

        train_str = 'validation' if not self.activity_net else 'training'    # Thumos and ActivityNet training set
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == train_str:
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):

        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i); break;
            self.classwiseidx.append(idx)


    def load_data(self, is_training=True):
        
        if is_training==True:
            features = []
            labels = []
            idx = []
            
            # random sampling
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            count_labels = np.array([self.count_labels[i] for i in idx])
            point_index = np.array([self.points2Idx[i] for i in idx])
            # print(point_index.shape)
            # if self.labels101to20 is not None:
            #     count_labels = count_labels[:,self.labels101to20]
            
            if self.t_max == -1:
                mx_len = 0
                # print(len(features))
                for i in idx:
                    # print(self.features[i].shape[0])
                    if mx_len < self.features[i].shape[0]:
                        mx_len = self.features[i].shape[0]
                # print(mx_len)

                self.t_max = mx_len
            

            pFeat = []
            pPoint = []
            pLabel = []
            # print('load data begin!!!')
            for k,i in enumerate(idx):
                # print(len(self.points2Idx[i]),len(self.labels_onehot[i]))
                length = len(self.points2Idx[i])
                # print(point_index[k])
                Res = utils.process_feat(self.features[i],self.points2Idx[i],self.labels_onehot[i][0:length], self.t_max)
                # print(Res[1])
                # input()
                # print(len(Res[1]) , len(Res[2]))
                # if len(Res[1]) != len(Res[2]):
                #     print(len(Res[1]) , len(Res[2]))
                pFeat.append(Res[0])
                pPoint.append(Res[2])
                pLabel.append(Res[1])
            # [ for i in idx]
            # print(self.t_max)
            # print(len(pFeat),len(pPoint),len(pL))
            # print(np.array([utils.process_feat(self.features[i], self.t_max) for i in idx]).shape)
            # print('load data over!!!')
            return np.array(pFeat), np.array([self.labels_multihot[i] for i in idx]),np.array(pLabel), count_labels,np.array(pPoint)

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array([feat]), np.array(labs), done



    def classes101to20(self):

        classlist20 = np.array([c.decode('utf-8') for c in self.classlist20])
        classlist101 = np.array([c.decode('utf-8') for c in self.classlist])
        labelsidx = []
        for categoryname in classlist20:
            labelsidx.append([i for i in range(len(classlist101)) if categoryname==classlist101[i]][0])
        
        return labelsidx


    def get_class_id(self):
        # Dict of class names and their indices
        d = dict()
        for i in range(len(self.classlist)):
            k = self.classlist[i]
            d[k.decode('utf-8')] = i
        return d


    def get_count(self):
        # Count number of instances of each category present in the video
        count = []
        num_class = len(self.class_order)
        for i in range(len(self.gtlabels)):
            gtl = self.gtlabels[i]
            cnt = np.zeros(num_class)
            for j in gtl:
                cnt[self.class_order[j]] += 1
            count.append(cnt)
        count = np.array(count)
        return count

