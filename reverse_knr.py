"""
Created on Thu Sep 20 21:34:40 2018
first commit: the error only back propogate one step back
second commit: the error can propogate all the way back to the first step, we do so by put all the calculation(including the KLMS part) inside the tensorflow

X is the composed of three successive data
@author: maying
"""
#import knr
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

# =============================================================================
# sequene reverse platform

N_tr=100000
N_te=100
Num_max=200 # maximum number of centers

def reverse_sequence_gen(L,TD=3,Dimx=5,L_max=20):
    data=np.random.randint(Dimx-1, size=(2*L+1))
    data_onehot=np.zeros((L_max,Dimx))
    for i in range(2*L+1):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=np.zeros((L_max,Dimx*TD))
    T=np.zeros((L_max,Dimx))
    X[0]=np.concatenate((np.zeros((1,2*Dimx)),data_onehot[0]), axis=None) 
    X[1]=np.concatenate((np.zeros((1,Dimx)), data_onehot[0],data_onehot[1]), axis=None) 
    for i in range(2,2*L+1):
        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+1):
            T[i]=data_onehot[i]
    for i in range (L+1,2*L+1):
            T[i]=data_onehot[2*L-i]  
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:2*L+1]=1
    return data_onehot,X,T,Mask

[data_onehot,X,T,MASK]=reverse_sequence_gen(3)
Dimd=np.shape(data_onehot)[1]
Dimx=np.shape(X)[1]
Dim=np.shape(T)[1]
L_max=20
#************************* data process begin******************************/
def checkequal(A,B):
    output=A-B
    output2=tf.reduce_sum(abs(output),axis=1)
    equality = tf.equal(output2, 0)
    output3=tf.cast(equality,tf.float32)
    output4=tf.expand_dims(output3,0)
    return output4

hiddenUnits = 10
tf.reset_default_graph()
data= tf.placeholder(tf.float32, [L_max,Dimd])
label = tf.placeholder(tf.float32, [L_max,Dim])
inputdata=tf.placeholder(tf.float32, [L_max,Dimx])
y=tf.placeholder(tf.float32, [L_max,Dim])
mask=tf.placeholder(tf.float32, [L_max,Dim])
M=tf.placeholder(tf.float32)

#L_effective=tf.placeholder(tf.int32)
#L_real=tf.placeholder(tf.int32)
Readweight=tf.placeholder(tf.float32)
Linkage=tf.placeholder(tf.float32)
#M_weight=tf.placeholder(tf.float32, [3,Num_max])
#alpha beta's RNN
weighti = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,Dimd]))
weighth = tf.Variable(tf.truncated_normal([hiddenUnits, hiddenUnits]))
biash = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,1]))
weight = tf.Variable(tf.truncated_normal([2,hiddenUnits]))  
bias = tf.Variable(tf.constant(0.1, shape=[2,1]))

weightpi = tf.Variable(tf.truncated_normal([3,hiddenUnits]))  
biaspi = tf.Variable(tf.constant(0.1, shape=[3,1]))
weightm = tf.Variable(tf.constant(0.1, shape=[Dim,Dimx]))
biasm = tf.Variable(tf.constant(0.1, shape=[Dim,1]))


hh=tf.zeros((hiddenUnits,1))
output_list = []
alpha = []
Pred=[]
Readweight_loop=Readweight
#KKK=M[0]-M
Link_after_loop=Linkage
Link_pre_loop=Linkage
Link_pre_loop_TEST=[]
Link_pre_NORM_TEST=[]
read1_TEST=[]
readx_TEST=[]
#Readweight0=checkequal(M,M[0])
Readweight1=Readweight_loop
for position in range(L_max):
    hh=tf.sigmoid(tf.matmul( weighti,tf.expand_dims(data[position],1)) + tf.matmul(weighth,hh) + biash)

    pi1=tf.matmul(weightpi,hh) + biaspi
    pi2=tf.reshape(pi1, [-1])
    pi3 = tf.nn.softmax(pi2)
#    pi3= tf.constant([[1.0],[0.0],[0.0]])
    pi4=tf.reshape(pi3, [3,1])
    alpha.append(pi4[0])
   #update Linkage
#    Readweight1=Readweight_loop    
    
    Readweight1=checkequal(M,inputdata[position])
    Readweight1x=checkequal(M,inputdata[2])
    if (position==0):
        Readweight2=Readweight_loop
        Readweight3=Readweight_loop
    else:
#        link1=tf.matmul(tf.expand_dims(Readweight1,0), tf.expand_dims(Readweight1,0), transpose_a=True) 
        Readx=checkequal(M,inputdata[position-1])
        Link_after_loop=Link_after_loop+tf.matmul(Readx,Readweight1, transpose_a=True) *alpha[position]
        Link_pre_loop=Link_pre_loop+tf.matmul(Readx, Readweight1, transpose_a=True) *alpha[position-1]
        Link_pre_loop_TEST.append(Link_pre_loop)
        read1_TEST.append(Readweight1)
        readx_TEST.append(Readx)
        Link_after_norm1=tf.reduce_sum(tf.maximum(Link_after_loop,1e-6),axis=1)
        Link_after_norm2=tf.maximum(Link_after_loop,1e-6)/tf.expand_dims(Link_after_norm1,1)
    
        Link_pre_norm1=tf.reduce_sum(tf.maximum(Link_pre_loop,1e-6),axis=0)
        Link_pre_norm2=tf.maximum(Link_pre_loop,1e-6)/Link_pre_norm1
  
        Readweight2=tf.matmul(Readweight_loop,Link_after_norm2, transpose_b=True)
        Readweight3=tf.matmul(Readweight_loop,Link_pre_norm2)
    
    Readweight123=tf.concat([Readweight1,Readweight2,Readweight3], axis=0)
# memory read function

    Readweight23=tf.matmul(tf.transpose(pi4[1:3]),Readweight123[1:3])
    Readvector23=tf.matmul(Readweight23,M)
    y2= tf.sigmoid(tf.matmul(weightm,tf.transpose(Readvector23)) + biasm)
    y2=tf.reshape(y2, [-1])
    pred =pi4[0]*y[position]+(pi4[1]+pi4[2])* y2   # have to change to y2 when the output dimension is corrected
#    pred =y[position] 
    Pred.append(pred)
    Readweight_loop=tf.matmul(tf.transpose(pi4),Readweight123)
Pred_final=tf.stack(Pred)
alpha_final=tf.stack(alpha)
error=label-Pred_final
loss=tf.losses.mean_squared_error(
    label,
    Pred_final,
    weights=mask,
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
mse_tr=np.zeros(N_tr)
j=0
#Centerincrease=np.zeros((Num_max))
Center=np.zeros((Num_max,Dimx))
Center_weight=np.zeros((Num_max,Dim))   

for i in range (N_tr):
    l=np.random.randint(4)+2
    [data_onehot,X,T,MASK]=reverse_sequence_gen(l)
    L=2*l+1
    e=np.zeros((L,Dim))
    y1=np.zeros((L_max,Dim))
#    minvalue=np.zeros(L)
#    minposition=np.zeros(L)
    X_shink=[]
    X_shink.append(X[0])
    le=1
    for ii in range (2*l+1):
        Ds = np.subtract(Center[0:j+1],X[ii])
        qVals = np.sum(Ds**2,axis=1)
#        minvalue[ii]=np.min(qVals)
#        minposition[ii]=np.argmin(qVals)
        qVals=np.expand_dims(qVals, axis=1)
        y1[ii] = np.sum(lr*Center_weight[:j+1]*(np.exp(-1*qVals/sigma)*np.ones((1,Dim))),axis=0)  
        
        Ds1 = np.subtract(np.asarray(X_shink),X[ii])
        qVals1 = np.sum(Ds1**2,axis=1)
        minvalue1=np.min(qVals1)
        if (minvalue1>th):
            X_shink.append(X[ii])
            le=le+1   
    Linkage_ini=np.zeros((le,le))
    Readweight_ini=np.ones((1,le))/le
    MM=np.asarray(X_shink)
#    Readweight1x_v=sess.run([checkequal(M,inputdata[2])],{data: data_onehot, label: T, y:y1,mask:MASK, M:MM,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})

    alpha_v,optimizer_v,error_v,pred_V,loss_v,Link_pre_loop_v,Link_pre_norm2_v,Link_pre_loop_TEST_v,read1_TEST_v,readx_TEST_v=sess.run([alpha_final,optimizer,error,Pred_final,loss,Link_pre_loop,Link_pre_norm2,Link_pre_loop_TEST,read1_TEST,readx_TEST], 
                                                       {data: data_onehot, label: T, y:y1,mask:MASK, M:MM,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})
    for ii in range (2*l+1):
        Ds = np.subtract(Center[0:j+1],X[ii])
        qVals = np.sum(Ds**2,axis=1)
        minvalue=np.min(qVals)
        minposition=np.argmin(qVals)
        if (minvalue>th):
            j=j+1
            Center[j]=X[ii]
            Center_weight[j]=error_v[ii]*alpha_v[ii]
            
#            Linkage[k][j]=alpha_v[ii]
#            k=j            
        else:
            Center_weight[minposition]=Center_weight[minposition]+error_v[ii]*alpha_v[ii]
#            Linkage[k][minposition]=alpha_v[ii]
#            k=minposition
#        Centerincrease[ii]=j
    mse_tr[i]=loss_v
    PRED=np.around(pred_V, decimals=1)
    if (i % 100 == 0):
        print (mse_tr[i])
    if (i % 1000 == 0):
        print ('predict',PRED,'target',T,mse_tr[i])
