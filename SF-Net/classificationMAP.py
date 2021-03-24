import numpy as np

def getAP(conf,labels):
    assert len(conf)==len(labels)
    sortind = np.argsort(-conf) # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 实际上ind按conf的值从大到小排列
    #print(sortind)
    tp = labels[sortind]==1; fp = labels[sortind]!=1
    #print('tp:{0}\nfp:{1}'.format(tp,fp))
    npos = np.sum(labels);

    #print('npos:{}'.format(npos))
    fp = np.cumsum(fp).astype('float32'); tp = np.cumsum(tp).astype('float32')  
    #numpy.cumsum(a, axis=None, dtype=None, out=None)
    #axis=0，按照行累加。
    #axis=1，按照列累加。
    #axis不给定具体值，就把numpy数组当成一个一维数组。
    #print('fp:{}'.format(fp))
    rec=tp/npos; prec=tp/(fp+tp)
    tmp = (labels[sortind]==1).astype('float32')

    #print(tmp)
    # if npos == 0:
    #     return 0

    return np.sum(tmp*prec)/npos

def getClassificationMAP(confidence,labels):
    ''' confidence and labels are of dimension n_samples x n_label '''

    AP = []
    # print(confidence.shape,labels.shape)
    for i in range(np.shape(labels)[1]-1):
       #print('confidece&labels for {}'.format(i))
       #print(confidence[:,i],'\n',labels[:,i])
       # print(confidence[:,i].shape,labels[:,i].shape)
       AP.append(getAP(confidence[:,i], labels[:,i]))
    return 100*sum(AP)/len(AP)
