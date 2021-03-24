import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()
        self.n_class = n_class
        n_featureby2 = int(n_feature / 2)
        outchal = n_featureby2 // 2
        self.base_f = nn.Sequential(
            nn.Conv1d(n_featureby2, outchal, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.base_r = nn.Sequential(
            nn.Conv1d(n_featureby2, outchal, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        # FC layers for the 2 streams
        n_featureby2 = outchal              # 512
        self.fc_f = nn.Sequential(
            nn.Linear(n_featureby2, n_featureby2),
            nn.LeakyReLU(),
            nn.Linear(n_featureby2, n_featureby2),
            nn.LeakyReLU()
        )
        
        self.fc_r = nn.Sequential(
            nn.Linear(n_featureby2, n_featureby2),
            nn.LeakyReLU(),
            nn.Linear(n_featureby2, n_featureby2),
            nn.LeakyReLU()
        )
        self.classifier_f = nn.Linear(n_featureby2, n_class)
        self.classifier_r = nn.Linear(n_featureby2, n_class)

        

        
        self.action_head_f = nn.Sequential(
            nn.Conv1d(n_featureby2,outchal,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_featureby2,outchal,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
        )
        self.action_bottom_f = nn.Linear(in_features=outchal,out_features=1)
        
        self.action_head_r = nn.Sequential(
            nn.Conv1d(n_featureby2,outchal,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_featureby2,outchal,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
        )
        self.action_bottom_r = nn.Linear(in_features=outchal,out_features=1)

        self.att_f = nn.Sequential(
            nn.Conv1d(n_featureby2,outchal,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_featureby2,1,kernel_size=1,stride=1,padding=0),
        )

        self.att_r = nn.Sequential(
            nn.Conv1d(n_featureby2,outchal,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_featureby2,1,kernel_size=1,stride=1,padding=0),
        )

        self.apply(weights_init)
        # Params for multipliers of TCams for the 2 streams
        self.mul_r = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.mul_f = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))

        self.act_mul_r = nn.Parameter(data=torch.Tensor(1).float().fill_(1))
        self.act_mul_f = nn.Parameter(data=torch.Tensor(1).float().fill_(1))

        self.att_mul_r = nn.Parameter(data=torch.Tensor(1).float().fill_(1))
        self.att_mul_f = nn.Parameter(data=torch.Tensor(1).float().fill_(1))

        self.dropout_f = nn.Dropout(0.7)
        self.dropout_r = nn.Dropout(0.7)
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=1)
        # self.category_r = nn.Conv2d(in_channels=1,out_channels=n_class,kernel_size=(1,1),stride=1)
        # self.category_f = nn.Conv2d(in_channels=1,out_channels=n_class,kernel_size=(1,1),stride=1)



    def forward(self, inputs, device='cpu', is_training=True, points_feature=None, seq_len=None):
        # inputs - batch x seq_len x featSize
        base_x_f = self.base_f(inputs[:, :, 1024:].permute(0, 2, 1)).permute(0, 2, 1)  # [24,750,1024]
        base_x_r = self.base_r(inputs[:, :, :1024].permute(0, 2, 1)).permute(0, 2, 1)

        if is_training:
            base_x_f = self.dropout_f(base_x_f)
            base_x_r = self.dropout_r(base_x_r)

        x_f = self.fc_f(base_x_f)
        x_r = self.fc_r(base_x_r)

        act_head_f = self.action_head_f(base_x_f.permute(0,2,1)).permute(0,2,1)
        act_head_r = self.action_head_r(base_x_r.permute(0,2,1)).permute(0,2,1)
        act_f = self.action_bottom_f(act_head_f)
        act_r = self.action_bottom_r(act_head_r)

        att_f = self.softmaxd1(self.att_f(base_x_f.permute(0,2,1)).permute(0,2,1))
        att_r = self.softmaxd1(self.att_r(base_x_r.permute(0,2,1)).permute(0,2,1))

        att_all = att_f * self.att_mul_f + att_r * self.att_mul_r

        act_all = act_r * self.act_mul_r + act_f * self.act_mul_f
        # att_f = self.softmaxd1(att_f)
        # att_r = self.softmaxd1(att_r)
        att_weight_all = F.softmax(att_all,dim=1)
        cls_x_f = self.classifier_f(x_f)
        cls_x_r = self.classifier_r(x_r)


        score_f = (cls_x_f[:,:,:-1] * att_f).sum(1)
        # print(global_score_f.shape)
        score_r = (cls_x_r[:,:,:-1]  * att_r).sum(1)

        tcam = cls_x_r * self.mul_r + cls_x_f * self.mul_f
        # print(tcam[:,:,:-1].shape,att_weight_all.shape)
        global_score = (tcam[:,:,:-1] * att_weight_all).sum(1)
        # print(tcam.shape)
        ### Add temporal conv for activity-net
        # if self.n_class == 100:
        #     tcam = self.relu(self.conv(tcam.permute([0, 2, 1]))).permute([0, 2, 1])

        count_feat = None
        if is_training:
            return cls_x_f, cls_x_r, tcam,score_f,score_r,global_score,base_x_f,base_x_r,torch.cat((base_x_r,base_x_f),2),act_f,act_r,act_all
        else:
            return x_f, cls_x_f[:,:,:-1], x_r, cls_x_r[:,:,:-1], tcam[:,:,:-1], count_feat