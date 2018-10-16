"""
Created on Thu Sep 20 21:34:40 2018
first commit: the error only back propogate one step back
second commit: the error can propogate all the way back to the first step, we do so by put all the calculation(including the KLMS part) inside the tensorflow
third commit: add another working mode: mode1: next means next in time step, model2: next means next useful input, the sequnece reverse problem can use model 1, 
              but the natural language processing problem should use model2
X is the composed of three successive data
@author: maying
"""
#import knr
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

# =============================================================================
# sequene reverse platform

N_tr=10000000
N_te=100
Num_max=1000 # maximum number of centers

def reverse_sequence_gen(L,TD=3,Dimx=5,L_max=25):
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
    aaa=2*L+1
    return data_onehot,X,T,Mask,aaa

def reverse_sequence_jump_gen(L,TD=3,Dimx=5,L_max=25):
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
    i=i-2
    aaa=np.int((L+1)/2)+L+1
    for j in range (L+1,aaa):
            T[j]=data_onehot[i]
            i=i-2
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:aaa]=1
    return data_onehot,X,T,Mask,aaa

def reverse_sequence_jump_gen_combine2(L,TD=2,Dimx=10,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+1))
    data_onehot=np.zeros((L_max,Dimx))
    for i in range(2*L+1):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=np.zeros((L_max,Dimx*TD))
    T=np.zeros((L_max,Dimx))
    
    data_COM=np.random.randint(Dimx-1, size=TD-1)
    data_onehot_COMPLEMENT=np.zeros((TD-1,Dimx))
#    data_norepeat=random.sample(range(Dimx-1), 2*L+1)
    for i in range(TD-1):
        data_onehot_COMPLEMENT[i][data_COM[i]]=1
    X[0]=np.concatenate((data_onehot_COMPLEMENT[0],data_onehot[0]), axis=None) 

#    X[0]=np.concatenate((np.zeros((1,2*Dimx)),data_onehot[0]), axis=None) 
#    X[1]=np.concatenate((np.zeros((1,Dimx)), data_onehot[0],data_onehot[1]), axis=None) 
    for i in range(1,2*L+1):
        X[i]=np.concatenate(( data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+1):
            T[i]=data_onehot[i]
    i=i-2
    aaa=np.int((L+1)/2)+L+1
    for j in range (L+1,aaa):
            T[j]=data_onehot[i]
            i=i-2
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:aaa]=1
    return data_onehot,X,T,Mask,aaa

def reverse_sequence_jump_gen_norepeat(L,TD=1,Dimx=100,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+1))
    data_onehot=np.zeros((L_max,Dimx))
    for i in range(2*L+1):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=data_onehot
    T=np.zeros((L_max,Dimx))
#    for i in range(1,2*L+1):
#        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+1):
            T[i]=data_onehot[i]
    i=i-2
    aaa=np.int((L+1)/2)+L+1
    for j in range (L+1,aaa):
            T[j]=data_onehot[i]
            i=i-2
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:aaa]=1
    return data_onehot,X,T,Mask,aaa


def reverse_sequence_jump3_gen_norepeat(L,TD=1,Dimx=100,L_max=25):
    repeat_times=int((L-1)/3)+1
    aaa=L+1+repeat_times
    data=np.random.randint(Dimx-1, size=(aaa))
    data_onehot=np.zeros((L_max,Dimx))
    for i in range(aaa):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=data_onehot
    T=np.zeros((L_max,Dimx))
#    for i in range(1,2*L+1):
#        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+1):
            T[i]=data_onehot[i]
#    i=i-2
#    aaa=np.int((L+1)/2)+L+1
    
    i=3*(repeat_times-1)
    for j in range (L+1,aaa):
            T[j]=data_onehot[i]
            i=i-3
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:aaa]=1
    return data_onehot,X,T,Mask,aaa


def reverse_sequence_norepeat_gen(L,TD=1,Dimx=100,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+1))
    data_onehot=np.zeros((L_max,Dimx))
#    data_norepeat=random.sample(range(Dimx-1), 2*L+1)
    for i in range(2*L+1):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=data_onehot
    T=np.zeros((L_max,Dimx))
    for i in range (L+1):
            T[i]=data_onehot[i]
    for i in range (L+1,2*L+1):
            T[i]=data_onehot[2*L-i]  
#    X=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:2*L+1]=1
    aaa=2*L+1
    return data_onehot,X,T,Mask,aaa



def reverse_sequence_gen_combinetwo(L,TD=2,Dimx=10,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+1))
    data_onehot=np.zeros((L_max,Dimx))
#    data_norepeat=random.sample(range(Dimx-1), 2*L+1)
    for i in range(2*L+1):
        data_onehot[i][data[i]]=1
    data_onehot[L][data[L]]=0
    data_onehot[L][Dimx-1]=1
    X=np.zeros((L_max,Dimx*TD))
    T=np.zeros((L_max,Dimx))
    data_COM=np.random.randint(Dimx-1, size=TD-1)
    data_onehot_COMPLEMENT=np.zeros((TD-1,Dimx))
#    data_norepeat=random.sample(range(Dimx-1), 2*L+1)
    for i in range(TD-1):
        data_onehot_COMPLEMENT[i][data_COM[i]]=1
    X[0]=np.concatenate((data_onehot_COMPLEMENT[0],data_onehot[0]), axis=None) 
#    X[1]=np.concatenate((np.zeros((1,Dimx)), data_onehot[0],data_onehot[1]), axis=None) 
    for i in range(1,2*L+1):
        X[i]=np.concatenate((data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+1):
            T[i]=data_onehot[i]
    for i in range (L+1,2*L+1):
            T[i]=data_onehot[2*L-i]  
#    X=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:2*L+1]=1
    aaa=2*L+1
    return data_onehot,X,T,Mask,aaa






def copy_sequence_gen(L,TD=3,Dimx=5,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+2))
    data_onehot=np.zeros((L_max,Dimx))
    data_onehot[0][Dimx-1]=1
    for i in range(1,2*L+2):
        data_onehot[i][data[i]]=1
    data_onehot[L+1][data[L+1]]=0
    data_onehot[L+1][Dimx-1]=1
    X=np.zeros((L_max,Dimx*TD))
    T=np.zeros((L_max,Dimx))
    X[0]=np.concatenate((data_onehot[L-1],data_onehot[L],data_onehot[0]), axis=None) 
    X[1]=np.concatenate((data_onehot[L], data_onehot[0],data_onehot[1]), axis=None) 
    for i in range(2,2*L+2):
        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+2):
            T[i]=data_onehot[i]
    for i in range (L+2,2*L+2):
            T[i]=data_onehot[i-L-1]  
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:2*L+2]=1
    aaa=2*L+2
    return data_onehot,X,T,Mask,aaa


def copy_sequence_norepeat_gen(L,TD=1,Dimx=100,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+2))
    data_onehot=np.zeros((L_max,Dimx))
    data_onehot[0][Dimx-1]=1
    for i in range(1,2*L+2):
        data_onehot[i][data[i]]=1
    data_onehot[L+1][data[L+1]]=0
    data_onehot[L+1][Dimx-1]=1
#    X=np.zeros((L_max,Dimx*TD))
    X=data_onehot
    T=np.zeros((L_max,Dimx))
#    X[0]=np.concatenate((data_onehot[L-1],data_onehot[L],data_onehot[0]), axis=None) 
#    X[1]=np.concatenate((data_onehot[L], data_onehot[0],data_onehot[1]), axis=None) 
#    for i in range(2,2*L+2):
#        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+2):
        T[i]=data_onehot[i]
    j=1
    for i in range (L+2,2*L+2):
        T[i]=data_onehot[j]
        j=j+1
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:2*L+2]=1
    aaa=2*L+2
    return data_onehot,X,T,Mask,aaa

def copy_sequence_jump_gen(L,TD=3,Dimx=5,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+2))
    data_onehot=np.zeros((L_max,Dimx))
    data_onehot[0][Dimx-1]=1
    for i in range(1,2*L+2):
        data_onehot[i][data[i]]=1
    data_onehot[L+1][data[L+1]]=0
    data_onehot[L+1][Dimx-1]=1
    X=np.zeros((L_max,Dimx*TD))
    T=np.zeros((L_max,Dimx))
    X[0]=np.concatenate((data_onehot[L-1],data_onehot[L],data_onehot[0]), axis=None) 
    X[1]=np.concatenate((data_onehot[L], data_onehot[0],data_onehot[1]), axis=None) 
    aaa=np.int((L+1)/2)+L+2
    for i in range(2,aaa):
        X[i]=np.concatenate((data_onehot[i-2], data_onehot[i-1],data_onehot[i]), axis=None) 
    for i in range (L+2):
        T[i]=data_onehot[i]
    i=1
    for j in range (L+2,aaa):
        T[j]=data_onehot[i]
        i=i+2
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:aaa]=1
    
    return data_onehot,X,T,Mask,aaa



import random
def copy_sequence_jump_norepeat_gen(L,TD=3,Dimx=100,L_max=25):
    data=np.random.randint(Dimx-1, size=(2*L+2))
    data_onehot=np.zeros((L_max,Dimx))
    data_onehot[0][Dimx-1]=1
#    data_norepeat=random.sample(range(Dimx-1), L+1)
    for i in range(1,L+1):
        data_onehot[i][data[i]]=1     
    for i in range(L+2,2*L+2):
        data_onehot[i][data[i]]=1 
#    data_onehot[L+1][data[L+1]]=0
    data_onehot[L+1][Dimx-1]=1
    X=data_onehot
#    np.zeros((L_max,Dimx*TD))
    T=np.zeros((L_max,Dimx))
    aaa=np.int((L+1)/2)+L+2
    for i in range (L+2):
        T[i]=data_onehot[i]
    i=1
    for j in range (L+2,aaa):
        T[j]=data_onehot[i]
        i=i+2
#    T=data_onehot
    Mask=np.zeros_like(T)
    Mask[0:aaa]=1
    
    return data_onehot,X,T,Mask,aaa





[data_onehot,X,T,MASK111,aaa111]=copy_sequence_jump_norepeat_gen(8)
Dimd=np.shape(data_onehot)[1]
Dimx=np.shape(X)[1]
Dim=np.shape(T)[1]
L_max=25
hiddenUnits = 10
mode=2
#gammaxxx= tf.constant([[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]])

#************************* data process begin******************************/
def checkequal(A,B):
    output=A-B
    output2=tf.reduce_sum(abs(output),axis=1)
    equality = tf.equal(output2, 0)
    output3=tf.cast(equality,tf.float32)
    output4=tf.expand_dims(output3,0)
    return output4


tf.reset_default_graph()
data= tf.placeholder(tf.float32, [L_max,Dimd])
label = tf.placeholder(tf.float32, [L_max,Dim])
inputdata=tf.placeholder(tf.float32, [L_max,Dimx])
y=tf.placeholder(tf.float32, [L_max,Dim])
mask=tf.placeholder(tf.float32, [L_max,Dim])
M=tf.placeholder(tf.float32)
#Inputdic=tf.placeholder(tf.float32)
gammaxxx= tf.constant([[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0],[0.0],[1.0]])
#L_effective=tf.placeholder(tf.int32)
#L_real=tf.placeholder(tf.int32)
Readweight=tf.placeholder(tf.float32)
Linkage=tf.placeholder(tf.float32)
#M_weight=tf.placeholder(tf.float32, [3,Num_max])
#alpha beta's RNN
weighti = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,Dimd]))
weighth = tf.Variable(tf.truncated_normal([hiddenUnits, hiddenUnits]))
biash = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,1]))

weightigamma = tf.Variable(tf.constant(0.5, shape=[hiddenUnits,Dimd]))
weighthgamma = tf.Variable(tf.truncated_normal([hiddenUnits, hiddenUnits]))
biashgamma = tf.Variable(tf.constant(0.5, shape=[hiddenUnits,1]))


weight = tf.Variable(tf.truncated_normal([2,hiddenUnits]))  
bias = tf.Variable(tf.constant(0.1, shape=[2,1]))

weightgamma= tf.Variable(tf.truncated_normal([1,hiddenUnits]))  
biasgamma = tf.Variable(tf.constant(0.1, shape=[1,1]))

weightpi = tf.Variable(tf.truncated_normal([3,hiddenUnits]))  
biaspi = tf.Variable(tf.constant(0.1, shape=[3,1]))
weightm = tf.Variable(tf.constant(0.1, shape=[Dim,Dimx]))
biasm = tf.Variable(tf.constant(0.1, shape=[Dim,1]))


hh=tf.zeros((hiddenUnits,1))
hhgamma=tf.zeros((hiddenUnits,1))
output_list = []
alpha = []
Pred=[]
gamma=[]
Readweight_loop=Readweight
#KKK=M[0]-M
Link_after_loop=Linkage
Link_pre_loop=Linkage
Link_after_loop_TEST=[]
Link_after_NORM_TEST=[]
Link_pre_loop_TEST=[]
Link_pre_NORM_TEST=[]
Readweight123_TEST=[]
Readweight_loop_TEST=[]
R_after_TEST=[]
y2_TEST=[]
#Readweight0=checkequal(M,M[0])
Readweight1=Readweight_loop

R_after=tf.constant([1.0])
#R_pre=tf.constant(1.0)
#R_after=[]
#contantone=tf.constant([1.0])
#R_after.append(contantone)
#R_after1=tf.unstack(R_after)
for position in range(L_max):
#for position in range(2):
    hh=tf.sigmoid(tf.matmul( weighti,tf.expand_dims(data[position],1)) + tf.matmul(weighth,hh) + biash)
    hhgamma=tf.sigmoid(tf.matmul( weightigamma,tf.expand_dims(data[position],1)) + tf.matmul(weighthgamma,hhgamma) + biashgamma)
    gamma1=tf.sigmoid(tf.matmul(weightgamma,hhgamma) + biasgamma)
    pi1=tf.matmul(weightpi,hh) + biaspi
    pi2=tf.reshape(pi1, [-1])
    pi3 = tf.nn.softmax(pi2)
#    pi3= tf.constant([[1.0],[0.0],[0.0]])
    pi4=tf.reshape(pi3, [3,1])
    alpha.append(pi4[0])
    gamma.append(gamma1)
   #update Linkage
#    Readweight1=Readweight_loop    
    R_pre=tf.constant(1.0)
    Readweight1=checkequal(M,inputdata[position])
#    Readweight1x=checkequal(M,inputdata[2])
    if (position==0):
        Readweight2=Readweight_loop
        Readweight3=Readweight_loop
    else:
        Link_after_norm1=tf.reduce_sum(tf.maximum(Link_after_loop,1e-6),axis=1)
        Link_after_norm2=tf.maximum(Link_after_loop,1e-6)/tf.expand_dims(Link_after_norm1,1)
    
        Link_pre_norm1=tf.reduce_sum(tf.maximum(Link_pre_loop,1e-6),axis=0)
        Link_pre_norm2=tf.maximum(Link_pre_loop,1e-6)/Link_pre_norm1
        
        Link_after_loop_TEST.append(Link_after_loop)
        Link_after_NORM_TEST.append(Link_after_norm2)
        
        Link_pre_loop_TEST.append(Link_pre_loop)
        Link_pre_NORM_TEST.append(Link_pre_norm2)
        R_after_TEST.append(R_after)
  
        Readweight2=tf.matmul(Readweight_loop,Link_after_norm2)
        Readweight3=tf.expand_dims(tf.reduce_sum(tf.matmul(tf.matrix_diag(Readweight_loop[0]),Link_pre_norm2, transpose_b=True),axis=0),0)
#        Readweight3=tf.reshape(tf.matmul(Link_pre_norm2,tf.expand_dims(Readweight_loop[0],1)), tf.shape(Readweight_loop))
        Readweight3=Readweight2
        
        if (mode==1):
            Readx=checkequal(M,inputdata[position-1])
            Link_after_loop=Link_after_loop+tf.matmul(Readx,Readweight1, transpose_a=True) *alpha[position]
            Link_pre_loop=Link_pre_loop+tf.matmul(Readx, Readweight1, transpose_a=True) *alpha[position-1]
        else:
#            memory_length=tf.maximum(position,10)
            for memory_i in range(1,position+1):
                if ((position-memory_i)<5):
                    Readx=checkequal(M,inputdata[memory_i-1])
                    Link_after_loop=Link_after_loop+tf.matmul(Readx,Readweight1, transpose_a=True) *alpha[position]*gamma[position]*R_after[memory_i-1]
#                    Link_after_loop=Link_after_loop+tf.matmul(Readx,Readweight1, transpose_a=True) *alpha[position]*R_after[memory_i-1]
                if (memory_i<5):
                    Ready=checkequal(M,inputdata[position-memory_i])
#                    Link_pre_loop=Link_pre_loop+tf.matmul(Ready, Readweight1, transpose_a=True) *alpha[position-memory_i]*gamma[position-memory_i]*R_pre
                    Link_pre_loop=Link_pre_loop+tf.matmul(Ready, Readweight1, transpose_a=True) *alpha[position-memory_i]*gamma[position-memory_i]*alpha[position]*gamma[position]*R_pre
                    R_pre=R_pre*(1-alpha[position-memory_i]*gamma[position-memory_i]*alpha[position]*gamma[position])
#                    R_pre=R_pre*(1-alpha[position-memory_i]*gamma[position-memory_i])
            R_after=R_after*(1-alpha[position]*gamma[position][0])
#            R_after=R_after*(1-alpha[position])
            R_after=tf.concat( [R_after, [1.0]],0)


    Readweight123=tf.concat([Readweight1,Readweight2,Readweight3], axis=0)
    Readweight123_TEST.append(Readweight123)
    Readweight23=tf.matmul(tf.transpose(pi4[1:3]),Readweight123[1:3])
    Readvector23=tf.matmul(Readweight23,M)
#    y2= tf.sigmoid(tf.matmul(weightm,tf.transpose(Readvector23)) + biasm)
    y2= Readvector23
    y2=tf.reshape(y2, [-1])
    y2_TEST.append(y2)
    pred =pi4[0]*y[position]+ y2   # have to change to y2 when the output dimension is corrected
#    pred =y[position] 
    Pred.append(pred)
    Readweight_loop=tf.matmul(tf.transpose(pi4),Readweight123)
    Readweight_loop_TEST.append(Readweight_loop)
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
print('1015')
for i in range (N_tr):
#    l1=np.random.randint(3)+2
#    l=2*l1
    l=np.random.randint(10)+2
    [data_onehot,X,T,MASK,Leffect]=copy_sequence_jump_norepeat_gen(l)
#    L=2*l+1
    e=np.zeros((Leffect,Dim))
    y1=np.zeros((L_max,Dim))
#    minvalue=np.zeros(L)
#    minposition=np.zeros(L)
    X_shink=[]
    X_shink.append(X[0])
    le=1
    for ii in range (Leffect):
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
#    gamma_v,gammaxxx_v,R_after_v=sess.run([gamma,gammaxxx,R_after],{data: data_onehot, label: T, y:y1,mask:MASK, M:MM,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})

    y2_TEST_v,R_after_TEST_v,Readweight_loop_TEST_v,Readweight123_TEST_v,Link_after_loop_TEST_v,gammaxxx_v,\
    Link_after_loop_v,Link_after_norm2_v,gamma_v,Link_pre_NORM_TEST_v,Link_pre_loop_TEST_v,\
    alpha_v,optimizer_v,error_v,pred_V,loss_v=\
    sess.run([y2_TEST,R_after_TEST,Readweight_loop_TEST,Readweight123_TEST,Link_after_loop_TEST,gammaxxx,\
    Link_after_loop,Link_after_norm2,gamma,Link_pre_NORM_TEST,Link_pre_loop_TEST,\
    alpha_final,optimizer,error,Pred_final,loss], \
    {data: data_onehot, label: T, y:y1,mask:MASK, M:MM,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})
#    alpha_v,optimizer_v,error_v,pred_V,loss_v=sess.run([alpha_final,optimizer,error,Pred_final,loss], 
#                                                       {data: data_onehot, label: T, y:y1,mask:MASK, M:MM,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})

    for ii in range (Leffect):
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
    if ((i % 3000 == 0)):
        print ('predict',PRED,'target',T)
        print (alpha_v,gamma_v)
#    if ((i>10000)&(mse_tr[i]>0.01)):
#        print ('predict',PRED,'target',T,mse_tr[i],i)
