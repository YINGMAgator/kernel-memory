#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:34:40 2018

@author: maying
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:46:13 2018
X is the composed of three successive data
@author: maying
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# =============================================================================
# sequene reverse platform

N_tr=20000
N_te=100
Num_max=40

def reverse_sequence_gen(L,TD=3,Dimx=5):
    data=np.random.randint(Dimx-1, size=(2*L+1))
    data_onehot=np.zeros((2*L+1,Dimx))
    for i in range(2*L+1):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=np.zeros((2*L+1,Dimx*TD))
    T=np.zeros((2*L+1,Dimx))
    X[0]=np.concatenate((np.zeros((1,2*Dimx)),data_onehot[0]), axis=None) 
    X[1]=np.concatenate((np.zeros((1,Dimx)), data_onehot[0],data_onehot[1]), axis=None) 
    for i in range(2,2*L+1):
        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+1):
            T[i]=data_onehot[i]
    for i in range (L+1,2*L+1):
            T[i]=data_onehot[2*L-i]  
#    T=data_onehot
    return data_onehot,X,T


[data_onehot,X,T]=reverse_sequence_gen(3)

Dimd=np.shape(data_onehot)[1]
Dimx=np.shape(X)[1]
Dim=np.shape(T)[1]

#Error_knr=knr.knr(X,T,X_te,T_te)
#print(Error_knr)
    
#************************* data process begin******************************/
hiddenUnits = 10
tf.reset_default_graph()
data= tf.placeholder(tf.float32, [Dimd])
label = tf.placeholder(tf.float32, [Dim])
y=tf.placeholder(tf.float32, [Dim])
M=tf.placeholder(tf.float32, [Num_max,Dimd])
M_weight=tf.placeholder(tf.float32, [3,Num_max])
#alpha beta's RNN
hstart=tf.placeholder(tf.float32, [hiddenUnits,1])
weighti = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,Dimd]))
weighth = tf.Variable(tf.truncated_normal([hiddenUnits, hiddenUnits]))
biash = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,1]))
weight = tf.Variable(tf.truncated_normal([2,hiddenUnits]))  
bias = tf.Variable(tf.constant(0.1, shape=[2,1]))

weightpi = tf.Variable(tf.truncated_normal([3,hiddenUnits]))  
biaspi = tf.Variable(tf.constant(0.1, shape=[3,1]))
#hh=tf.zeros((hiddenUnits,1))
hh=hstart
#    output_list = []
#for position in range(Lstatemax):
hh=tf.sigmoid(tf.matmul( weighti,tf.expand_dims(data,1)) + tf.matmul(weighth,hh) + biash)
hend=hh
alphabeta1=tf.matmul(weight,hh) + bias
alphabeta2=tf.reshape(alphabeta1, [-1])
alphabeta = tf.nn.softmax(alphabeta2)
alpha=alphabeta[0]
beta=alphabeta[1]
#beta= tf.constant(1.0)
#alpha= tf.constant(0.0)
pi1=tf.matmul(weightpi,hh) + biaspi
pi2=tf.reshape(pi1, [-1])
pi3 = tf.nn.softmax(pi2)
pi4=tf.reshape(pi3, [3,1])
#pi = tf.constant([[0.0],[0.0],[1.0]])
# memory read function
weightm = tf.Variable(tf.constant(0.1, shape=[Dim,Dimd]))
biasm = tf.Variable(tf.constant(0.1, shape=[Dim,1]))
#y2= (tf.matmul(weightm,tf.expand_dims(data,1)) + biasm)

Readweight=tf.matmul(tf.transpose(pi4),M_weight)
Readvector=tf.matmul(Readweight,M)
y2= tf.sigmoid(tf.matmul(weightm,tf.transpose(Readvector)) + biasm)
#y2= (tf.matmul(weightm,tf.expand_dims(data,1)) + biasm)
y2=tf.reshape(y2, [-1])
pred =alpha*y+beta* y2   # have to change to y2 when the output dimension is corrected



error=label-pred
loss=tf.losses.mean_squared_error(
    label,
    pred,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
)
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# train
lr=0.1
sigma=1

th=0.5
Lstatemax=10
mse_tr=np.zeros(N_tr)


for i in range (1,N_tr):
    l=np.random.randint(4)+2
    [data_onehot,X,T]=reverse_sequence_gen(l)
    L=2*l+1
    mse=np.zeros(L)  
    PRED=np.zeros((L,Dim))  
    
    
    e=np.zeros((L,Dim))
    y1=np.zeros((L,Dim))

    Centerincrease=np.zeros((L))
    Center=np.zeros((Num_max,np.shape(X)[1]))
    Center_weight=np.zeros((Num_max,Dim))   
    
    Memory=np.zeros((Num_max,Dimd)) 
    Memory[0]=data_onehot[0]

    Memory_weight=np.zeros((3,Num_max)) 
    Memory_weight[0][0]=1
    Memory_weight[1][0]=1
    Memory_weight[2][0]=1
      
    hiddenstate=np.zeros((hiddenUnits,1))
    Alpha=np.zeros(N_tr)
#    inputdata = np.zeros((Lstatemax,np.shape(X)[1]))    
    y1[0]=np.zeros(Dim)   

    Readweight_v,alpha_v,optimizer_v,error_v,hend_v,pred_V,loss_v=sess.run([Readweight,alpha,optimizer,
                error,hend,pred,loss], {data: data_onehot[0], label: T[0], y:y1[0], M:Memory, M_weight:Memory_weight, hstart:hiddenstate})
    hiddenstate=hend_v
    Alpha[0]=alpha_v
    e[0]=error_v
    PRED[0]=np.around(pred_V, decimals=1)
    mse[0]=np.sum(e[0]**2)  
    Center[0]=X[0]
    Center_weight[0]=e[0]*Alpha[0]
    j=0
    #memory controller matrix
    Linkage = np.zeros((Num_max,Num_max)) 
    k=0
    for ii in range (1,2*l+1):
        Ds = np.subtract(Center[0:ii],X[ii])
        qVals = np.sum(Ds**2,axis=1)
        minvalue=np.min(qVals)
        minposition=np.argmin(qVals)
        qVals=np.expand_dims(qVals, axis=1)
        y1[ii] = np.sum(lr*Center_weight[:ii]*(np.exp(-1*qVals/sigma)*np.ones((1,Dim))),axis=0)  
        rLL = np.transpose(np.sum(Linkage[:j+1,:j+1],axis=1)*np.ones((j+1,j+1)))
        rLinkagenorm1=Linkage[:j+1,:j+1]+(rLL==0).astype(int)
        rLinkagenorm2=np.zeros_like(Linkage,dtype=float)
        rLinkagenorm2[:j+1,:j+1]=rLinkagenorm1/np.transpose(np.sum(rLinkagenorm1,axis=1)*np.ones((j+1,j+1)))
        
        cLL = (np.sum(Linkage[:j+1,:j+1],axis=0)*np.ones((j+1,j+1)))
        cLinkagenorm1=Linkage[:j+1,:j+1]+(cLL==0).astype(int)
        cLinkagenorm2=np.zeros_like(Linkage,dtype=float)
        cLinkagenorm2[:j+1,:j+1]=cLinkagenorm1/np.sum(cLinkagenorm1,axis=0)*np.ones((j+1,j+1))
           
        Memory_weight=np.zeros((3,Num_max)) 
        Memory_weight[0]=np.matmul(Readweight_v,rLinkagenorm2)
        Memory_weight[1]=np.matmul(Readweight_v,np.transpose(cLinkagenorm2))
        if minvalue>th:
            j=j+1
            Memory_weight[2][j]=1
            Memory[j]=data_onehot[ii]
        else:
            Memory_weight[2][minposition]=1
            
        
        Readweight_v,alpha_v,optimizer_v,error_v,hend_v,pred_V,loss_v=sess.run([Readweight,alpha,optimizer,
                             error,hend,pred,loss], {data: data_onehot[ii], label: T[ii], y:y1[ii], M:Memory, M_weight:Memory_weight, hstart:hiddenstate})
        hiddenstate=hend_v
        Alpha[ii]=alpha_v
        e[ii]=error_v
        PRED[ii]=np.around(pred_V, decimals=1)
        mse[ii]=np.sum(e[ii]**2)
        if minvalue>th:
#            j=j+1
            Center[j]=X[ii]
            Center_weight[j]=e[ii]*Alpha[ii]
            
            Linkage[k][j]=Alpha[ii]
            k=j            
        else:
            Center_weight[minposition]=Center_weight[minposition]+e[ii]*Alpha[ii]
            Linkage[k][minposition]=Alpha[ii]
            k=minposition
        Centerincrease[ii]=j
    mse_tr[i]=np.sum(mse)
    if (i % 100 == 0):
        print ('predict',PRED,'target',T,mse_tr[i])
