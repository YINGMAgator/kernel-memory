import KAARMA
import klms
import qklms
#import knr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# =============================================================================
# # TOMITA test platform
# from tomita import trainlab,strain,testlab,stest,mask,maskv
# KAARMA_TOMITA=KAARMA.KAARMA(strain,trainlab,mask,stest,testlab,maskv)
# KAARMA_TOMITA.train()
# =============================================================================
## Lorenz sequene platform
#
##data preparation
#import scipy.io
#mat = scipy.io.loadmat('Lorenz_normal.mat') 
#Lorenz=mat['Lorenz']
#noise_power=0.1
#Lorenz = Lorenz + noise_power*np.random.randn( np.shape(Lorenz)[0], np.shape(Lorenz)[1])
#
##parameters setting
#componenti=0
#componento=0
#TD=10
#shiftrange=1000
#N_tr=5000
#N_te=1000
#N_total=N_tr+N_te+shiftrange
#X_org=np.zeros((N_total,TD))
#for i in range(N_total):
#    X_org[i]=Lorenz[componenti][i:i+TD]
#T_org=Lorenz[componento][TD:TD+N_total]
#T_org = np.expand_dims(T_org, axis=2)
#print ('using compoent', componenti, 'to predict', componento)
#shiftstep=np.random.randint(shiftrange)
#X=X_org[shiftstep:shiftstep+N_tr]
#T=T_org[shiftstep:shiftstep+N_tr]
#
#X_te=X_org[shiftstep+N_tr:shiftstep+N_tr+N_te]
#T_te=T_org[shiftstep+N_tr:shiftstep+N_tr+N_te]
#
#
#
###KLMS
##Error_klms=klms.klms(X,T,X_te,T_te)
###QKLMS
##Error_qklms=qklms.qklms(X,T,X_te,T_te)
###QKLMS
##Error_knr=knr.knr(X,T,X_te,T_te)
##print(Error_knr)
#
#lr=0.1
#sigma=1
#Num_max=2000
#th=0.5
#Lstatemax=10
#N_tr=np.shape(X)[0]
#N_te=np.shape(X_te)[0]
#Dimx=np.shape(X)[1]
#Dim=np.shape(T)[1]
#e=np.zeros((N_tr,Dim))
#y1=np.zeros((N_tr,Dim))
#mse=np.zeros(N_tr)
#mse_te=np.zeros(N_te)
#
#Centerincrease=np.zeros((N_tr))
#
#Center=np.zeros((Num_max,np.shape(X)[1]))
#Center_weight=np.zeros((Num_max,Dim))
#    
##************************* data process begin******************************/
#hiddenUnits = 2
#tf.reset_default_graph()
#data= tf.placeholder(tf.float32, [Lstatemax, Dimx])
#label = tf.placeholder(tf.float32, [Dim])
#y=tf.placeholder(tf.float32, [Dim])
#
##alpha beta's RNN
#hstart=tf.placeholder(tf.float32, [hiddenUnits,1])
#weighti = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,Dimx]))
#weighth = tf.Variable(tf.truncated_normal([hiddenUnits, hiddenUnits]))
#biash = tf.Variable(tf.constant(0.1, shape=[hiddenUnits,1]))
#weight = tf.Variable(tf.truncated_normal([2,hiddenUnits]))  
#bias = tf.Variable(tf.constant(0.1, shape=[2,1]))
#biastest = tf.Variable(tf.constant(0.1, shape=[2]))
##hh=tf.zeros((hiddenUnits,1))
#hh=hstart
##    output_list = []
#for position in range(Lstatemax):
#    hh=tf.sigmoid(tf.matmul( weighti,tf.expand_dims(data[position],1)) + tf.matmul(weighth,hh) + biash)
#hend=hh
#alphabeta = tf.nn.softmax(tf.matmul(weight,hh) + bias)
#alpha=alphabeta[0]
#beta=alphabeta[1]
##beta= tf.constant(1.0)
##alpha= tf.constant(0.0)
#
## memory read function
#weightm = tf.Variable(tf.constant(0.1, shape=[Dim,Dimx]))
#biasm = tf.Variable(tf.constant(0.1, shape=[Dim,1]))
#y2= (tf.matmul(weightm,tf.expand_dims(data[position],1)) + biasm)
#pred =alpha*y+beta* y2[0]   # have to change to y2 when the output dimension is corrected
#
#
#
#error=label-pred
#loss=tf.losses.mean_squared_error(
#    label,
#    pred,
#    scope=None,
#    loss_collection=tf.GraphKeys.LOSSES,
#)
#optimizer = tf.train.AdamOptimizer().minimize(loss)
#
#
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#
## train
#e[0]=T[0]
#y1[0]=np.zeros(Dim)
#mse[0]=np.sum(T[0]**2)
#Center[0]=X[0]
#Center_weight[0]=e[0]
#j=1
#Alpha=np.zeros(N_tr)
#inputdata = np.zeros((Lstatemax,np.shape(X)[1]))
#hiddenstate=np.zeros((hiddenUnits,1))
#
#PRED=np.zeros(N_tr)
#for i in range (1,N_tr):
#    Ds = np.subtract(Center[0:i],X[i])
#    qVals = np.sum(Ds**2,axis=1)
#    minvalue=np.min(qVals)
#    minposition=np.argmin(qVals)
#    qVals=np.expand_dims(qVals, axis=1)
#    y1[i] = np.sum(lr*Center_weight[:i]*(np.exp(-1*qVals/sigma)*np.ones((1,Dim))),axis=0)
#    
#   
#    inputdata[0:Lstatemax-1]=inputdata[1:Lstatemax]              
#    inputdata[Lstatemax-1]=X[i]  
#    alpha_v,optimizer_v,error_v,hend_v,pred_V=sess.run([alpha,optimizer,error,hend,pred], {data: inputdata, label: T[i], y:y1[i], hstart:hiddenstate})
#    hiddenstate=hend_v
#    Alpha[i]=alpha_v
#    e[i]=error_v
#    PRED[i]=pred_V
##    e[i]=T[i]-y1[i]
#    mse[i]=np.sum(e[i]**2)
#    if minvalue>th:
#        Center[j]=X[i]
#        Center_weight[j]=e[i]*Alpha[i]
#        j=j+1
##            print("iteration",i,"center number",j)
#    else:
#        Center_weight[minposition]=Center_weight[minposition]+e[i]*Alpha[i]
#    Centerincrease[i]=j
#
#n=np.arange(N_tr)
#plt.close()
#plt.ion()
#plt.subplot(3,1,1)
#plt.plot(n,T,n,PRED,'r')
#plt.title('train')  
##n=np.arange(N_te)
##plt.subplot(3,1,2)
##plt.plot(n,T_te,n,y_te,'r')
##plt.title('test')
#plt.subplot(3,1,3)
#plt.plot(e)
#plt.title('center number')
##plt.subplot(3,1,3)
##plt.plot(Centerincrease)
##plt.title('center number')
#plt.show()
#plt.draw()

# =============================================================================
# sequene reverse platform

#data preparation
#import scipy.io
#mat = scipy.io.loadmat('Lorenz_normal.mat') 
#Lorenz=mat['Lorenz']
#noise_power=0.1
#Lorenz = Lorenz + noise_power*np.random.randn( np.shape(Lorenz)[0], np.shape(Lorenz)[1])
#
##parameters setting
#componenti=0
#componento=0
#TD=10
#shiftrange=1000
N_tr=10000
N_te=100
Num_max=40
#N_total=N_tr+N_te+shiftrange
#X_org=np.zeros((N_total,TD))
#for i in range(N_total):
#    X_org[i]=Lorenz[componenti][i:i+TD]
#T_org=Lorenz[componento][TD:TD+N_total]
#T_org = np.expand_dims(T_org, axis=2)
#print ('using compoent', componenti, 'to predict', componento)
#shiftstep=np.random.randint(shiftrange)
#X=X_org[shiftstep:shiftstep+N_tr]
#T=T_org[shiftstep:shiftstep+N_tr]
#X_te=X_org[shiftstep+N_tr:shiftstep+N_tr+N_te]
#T_te=T_org[shiftstep+N_tr:shiftstep+N_tr+N_te]

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
##KLMS
#Error_klms=klms.klms(X,T,X_te,T_te)
##QKLMS
#Error_qklms=qklms.qklms(X,T,X_te,T_te)
##QKLMS
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
#mse_te=np.zeros(N_te)


for i in range (1,N_tr):
    l=np.random.randint(4)+2
    [data_onehot,X,T]=reverse_sequence_gen(l)
    L=2*l+1
    e=np.zeros((L,Dim))
    y1=np.zeros((L,Dim))

    Centerincrease=np.zeros((L))
    Center=np.zeros((Num_max,np.shape(X)[1]))
    Center_weight=np.zeros((Num_max,Dim))   
    
    Memory=np.zeros((Num_max,Dimd))
#    R_weight=np.zeros(Num_max)  
    Memory[0]=data_onehot[0]

    Memory_weight=np.zeros((3,Num_max)) 
    Memory_weight[0][0]=1
    Memory_weight[1][0]=1
    Memory_weight[2][0]=1
    
    mse=np.zeros(L)    
    hiddenstate=np.zeros((hiddenUnits,1))
    Alpha=np.zeros(N_tr)
#    inputdata = np.zeros((Lstatemax,np.shape(X)[1]))    
    PRED=np.zeros((L,Dim))  
    y1[0]=np.zeros(Dim)   

    Readweight_v,alpha_v,optimizer_v,error_v,hend_v,pred_V,loss_v=sess.run([Readweight,alpha,optimizer,
                error,hend,pred,loss], {data: data_onehot[0], label: T[0], y:y1[0], M:Memory, M_weight:Memory_weight, hstart:hiddenstate})
    hiddenstate=hend_v
    Alpha[0]=alpha_v
    e[0]=error_v
    PRED[0]=np.around(pred_V, decimals=1)
#    e[i]=T[i]-y1[i]
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
#        inputdata[0:Lstatemax-1]=inputdata[1:Lstatemax]              
#        inputdata[Lstatemax-1]=X[i]  
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
#        alpha_v=sess.run(data, {data: X[ii]})
        hiddenstate=hend_v
        Alpha[ii]=alpha_v
        e[ii]=error_v
        PRED[ii]=np.around(pred_V, decimals=1)
    #    e[i]=T[i]-y1[i]
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
    if (i % 100 == 0):
        print ('predict',PRED,'target',T,loss_v)
