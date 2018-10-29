#first version: works for task1, however, gamma does not converge to 0/1

#************************* data process begin******************************/
import os
import re
import numpy as np
#************************* data process begin******************************/
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

output_symbol = "-"
newstory_delimiter = " NEWSTORY "
tasks_dir = os.path.join("tasks_1-20_v1-2", "en-10k")
files_path = []
for f in os.listdir(tasks_dir):
    f_path = os.path.join(tasks_dir, f)
    if os.path.isfile(f_path):
        files_path.append(f_path)
all_input_stories, all_output_stories, all_masks = dict(), dict(), dict()
worddic=[]
for file_path1 in files_path:
    print(file_path1)
    file = open(file_path1).read().lower()
    file = re.sub("\n1 ", newstory_delimiter, file)  # adding a delimeter between two stories
    file = re.sub("\d+|\n|\t", " ", file)  # removing all numbers, newlines and tabs
    file = re.sub("([?.])", r" \1", file)  # adding a space before all punctuations
    stories = file.split(newstory_delimiter)
    for i, story in enumerate(stories):
        input_tokens = story.split()
        worddic.extend(input_tokens)
        worddic=set(worddic)
        worddic=list(worddic)
print (len(worddic))
# pdb.set_trace()
worddic.extend(output_symbol)   #extend the word dictionary with a output symbol "-"
print (len(worddic))
########################OneHotEncoder##########################
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(worddic)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#inverted = label_encoder.inverse_transform(np.argmax(onehot_encoded, axis=0))
#aaa=['mary','south']
#bbb=label_encoder.transform(aaa)
#bbb = bbb.reshape(len(bbb), 1)
#bbb=onehot_encoder.transform(bbb)
 

#************************* data preparation end ******************************/
output_symbol = "-"
newstory_delimiter = " NEWSTORY "
# processed_append = "-processed.p"
tasks_dir = os.path.join("tasks_1-20_v1-2", "en-10k")
files_path = []
for f in os.listdir(tasks_dir):
    f_path = os.path.join(tasks_dir, f)
    if os.path.isfile(f_path):
        files_path.append(f_path)
all_input_stories, all_output_stories, all_masks = dict(), dict(), dict()
for file_path1 in files_path:
    print(file_path1)
    file = open(file_path1).read().lower()
    file = re.sub("\n1 ", newstory_delimiter, file)  # adding a delimeter between two stories
    file = re.sub("\d+|\n|\t", " ", file)  # removing all numbers, newlines and tabs
    file = re.sub("([?.])", r" \1", file)  # adding a space before all punctuations
    stories = file.split(newstory_delimiter)
    # print (stories[0])
    input_stories = []
    output_stories = []
    masks=[]
    for i, story in enumerate(stories):
        input_tokens = story.split()
        output_tokens = story.split()
#        mask = [[0]] * len(output_tokens)
        for i, token in enumerate(input_tokens):
            if token == "?":
                input_tokens[i + 1] = output_symbol
#                mask [i+1] =[1]
#        input_tokens.remove('?')
#        output_tokens.remove('?')
        input_tokens = list(filter(lambda x: x!= '?', input_tokens))
        output_tokens = list(filter(lambda x: x!= '?', output_tokens))
        mask = [[1]] * len(output_tokens)
        for i, token in enumerate(input_tokens):
            if token == "-":
#                input_tokens[i + 1] = output_symbol
                mask [i] =[20]
        xx=label_encoder.transform(input_tokens)
        input_stories.append(xx)
        xx=label_encoder.transform(output_tokens)
        output_stories.append(xx)
        masks.append(mask)
        # print (input_stories)
        # print (label_encoder.transform(input_tokens))
        # print('input',input_tokens)
        # print ('output',output_stories)
        # print ('mask',masks)
        # pdb.set_trace()
    all_input_stories[file_path1] = np.asarray(input_stories)
    all_output_stories[file_path1] = np.asarray(output_stories)
    all_masks[file_path1] = np.asarray(masks)


from natsort import natsorted
train_list = natsorted([k for k, v in all_input_stories.items() if k[-9:] == "train.txt"])
test_list = natsorted([k for k, v in all_input_stories.items() if k[-8:] == "test.txt"])

# print(train_list[qa_index])
# # print(test_list[qa_index])
# print (all_input_stories[train_list[qa_index]].shape)
# print (all_input_stories[test_list[qa_index]].shape)
# print (all_input_stories[train_list[qa_index]][0])
# print (all_input_stories[train_list[qa_index]])
# pdb.set_trace()
#************************* data preparation end ******************************/
# rnn model

qa_index=0
#maxSeqLength = 2000
numDimensions = 54 #Dimensions for each word vector
N_tr=100000
N_te=200
batchSize=1
maxSeqLength=0
for i in range (len(all_input_stories[train_list[qa_index]])):
    maxSeqLength=max(maxSeqLength,len(all_input_stories[train_list[qa_index]][i]))
#************************* training process begin******************************/
    
#%%
import tensorflow as tf
from random import randint

def getTrainBatch(num,L_increase):
    arr = np.zeros([batchSize, maxSeqLength,numDimensions], dtype=int)
    labels = np.zeros([batchSize, maxSeqLength,numDimensions], dtype=int)
    mask = np.zeros([batchSize, maxSeqLength,numDimensions], dtype=int)
    for i in range(batchSize):
#        num = randint(1,N_tr-1)
        L=min(len(all_masks[train_list[qa_index]][num]),L_increase)
        bbb=all_input_stories[train_list[qa_index]][num]
        bbb = bbb.reshape(len(bbb), 1)
        # pdb.set_trace()
        arr[i][0:len(all_input_stories[train_list[qa_index]][num])] = onehot_encoder.transform(bbb)

        bbb=all_output_stories[train_list[qa_index]][num]
        bbb = bbb.reshape(len(bbb), 1)
        labels[i][0:len(all_output_stories[train_list[qa_index]][num])]=onehot_encoder.transform(bbb)

        mask[i][0:L] =all_masks[train_list[qa_index]][num][0:L]
       
    return arr[0],arr[0], labels[0],mask[0],L

def getTestBatch():
    arr = np.zeros([batchSize, maxSeqLength,numDimensions], dtype=int)
    labels = np.zeros([batchSize, maxSeqLength,numDimensions], dtype=int)
    mask = np.zeros([batchSize, maxSeqLength,numDimensions], dtype=int)
    for i in range(batchSize):
        num = randint(1,N_te-1)
        bbb=all_input_stories[test_list[qa_index]][num]
        bbb = bbb.reshape(len(bbb), 1)
        arr[i][0:len(all_input_stories[test_list[qa_index]][num])] =onehot_encoder.transform(bbb)
        bbb=all_output_stories[test_list[qa_index]][num]
        bbb = bbb.reshape(len(bbb), 1)
        labels[i][0:len(all_output_stories[test_list[qa_index]][num])]=onehot_encoder.transform(bbb)
        mask[i][0:len(all_masks[test_list[qa_index]][num])] =all_masks[test_list[qa_index]][num]
        L=len(all_masks[test_list[qa_index]][num])
    return arr[0], arr[0],labels[0],mask[0],L



[data_onehot,X,T,M1,aaa111]=getTrainBatch(0,10)

Dimd=np.shape(data_onehot)[1]
Dimx=np.shape(X)[1]
Dim=np.shape(T)[1]
L_max=maxSeqLength
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

weightiforgetcoef = tf.Variable(tf.constant(0.5, shape=[hiddenUnits,Dimd]))
weighthforgetcoef = tf.Variable(tf.truncated_normal([hiddenUnits, hiddenUnits]))
biashforgetcoef = tf.Variable(tf.constant(0.5, shape=[hiddenUnits,1]))


weight = tf.Variable(tf.truncated_normal([2,hiddenUnits]))  
bias = tf.Variable(tf.constant(0.1, shape=[2,1]))

weightgamma= tf.Variable(tf.truncated_normal([1,hiddenUnits]))  
biasgamma = tf.Variable(tf.constant(0.1, shape=[1,1]))

weightforgetcoef= tf.Variable(tf.truncated_normal([1,hiddenUnits]))  
biasforgetcoef = tf.Variable(tf.constant(0.1, shape=[1,1]))


weightpi = tf.Variable(tf.truncated_normal([3,hiddenUnits]))  
biaspi = tf.Variable(tf.constant(0.1, shape=[3,1]))
weightm = tf.Variable(tf.constant(0.1, shape=[Dim,Dimx]))
biasm = tf.Variable(tf.constant(0.1, shape=[Dim,1]))


hh=tf.zeros((hiddenUnits,1))
hhgamma=tf.zeros((hiddenUnits,1))
hhforgetcoef=tf.zeros((hiddenUnits,1))
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
forget_index_TEST=[]
forgetcoef_TEST=[]
Link_after_loop_inner_TEST=[]
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

    hhforgetcoef=tf.sigmoid(tf.matmul( weightiforgetcoef,tf.expand_dims(data[position],1)) + tf.matmul(weighthforgetcoef,hhforgetcoef) + biashforgetcoef)
    forgetcoef=tf.sigmoid(tf.matmul(weightforgetcoef,hhforgetcoef) + biasforgetcoef)    
    
    
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
    forget_index=tf.constant(1.0)
#    Readweight1x=checkequal(M,inputdata[2])
    
    if (position==0):
        Readweight2=Readweight_loop
        Readweight3=Readweight_loop
    else:
        Link_after_norm1=tf.reduce_sum(tf.maximum(Link_after_loop,1e-6),axis=1)
        Link_after_norm2=tf.maximum(Link_after_loop,1e-6)/tf.expand_dims(Link_after_norm1,1)
    
        Link_pre_norm1=tf.reduce_sum(tf.maximum(Link_pre_loop,1e-6),axis=0)
        Link_pre_norm2=tf.maximum(Link_pre_loop,1e-6)/Link_pre_norm1
        
        Link_after_loop_TEST.append(Link_after_loop_inner_TEST)
        Link_after_NORM_TEST.append(Link_after_norm2)
        Link_after_loop_inner_TEST=[]
        Link_pre_loop_TEST.append(Link_pre_loop)
        Link_pre_NORM_TEST.append(Link_pre_norm2)
        R_after_TEST.append(R_after)
  
        Readweight2=tf.matmul(Readweight_loop,Link_after_norm2)
        Readweight3=tf.expand_dims(tf.reduce_sum(tf.matmul(tf.matrix_diag(Readweight_loop[0]),Link_pre_norm2, transpose_b=True),axis=0),0)
#        Readweight3=tf.reshape(tf.matmul(Link_pre_norm2,tf.expand_dims(Readweight_loop[0],1)), tf.shape(Readweight_loop))
#        Readweight3=Readweight2
        
        if (mode==1):
            Readx=checkequal(M,inputdata[position-1])
            
            Link_after_loop=Link_after_loop+tf.matmul(Readx,Readweight1, transpose_a=True) *alpha[position]
            Link_pre_loop=Link_pre_loop+tf.matmul(Readx, Readweight1, transpose_a=True) *alpha[position-1]
        else:
            forget_index=tf.minimum(tf.reduce_sum(checkequal(inputdata[:position],inputdata[position])),1)
            matrixforget_index=tf.matmul(1-Readweight1*forget_index*forgetcoef,tf.ones_like(Readweight1),transpose_a=True)
            Link_after_loop=tf.multiply(Link_after_loop,matrixforget_index)
            matrixforget_index=tf.matmul(tf.ones_like(Readweight1),1-Readweight1*forget_index*forgetcoef,transpose_a=True)
            Link_after_loop=tf.multiply(Link_after_loop,matrixforget_index)
#            
            for memory_i in range(1,position+1):
                if ((position-memory_i)<6):
                    
                    Readx=checkequal(M,inputdata[memory_i-1])
                    position_maskx=tf.matmul(Readx,Readweight1, transpose_a=True)
                    Link_after_loop=Link_after_loop+position_maskx*alpha[position]*gamma[position]*R_after[memory_i-1]
#                    Link_after_loop=Link_after_loop+tf.matmul(Readx,Readweight1, transpose_a=True) *alpha[position]*R_after[memory_i-1]
                if (memory_i<6):
                    forget_index=tf.minimum(tf.reduce_sum(checkequal(inputdata[:position],inputdata[position])),1)
                    Ready=checkequal(M,inputdata[position-memory_i])
                    position_masky=tf.matmul(Ready, Readweight1, transpose_a=True)
                    Link_pre_loop=Link_pre_loop+position_masky *alpha[position-memory_i]*gamma[position-memory_i]*R_pre
#                    Link_pre_loop=Link_pre_loop+position_masky *alpha[position-memory_i]*gamma[position-memory_i]*alpha[position]*gamma[position]*R_pre
#                    R_pre=R_pre*(1-alpha[position-memory_i]*gamma[position-memory_i]*alpha[position]*gamma[position])
                    R_pre=R_pre*(1-alpha[position-memory_i]*gamma[position-memory_i])
                Link_after_loop_inner_TEST.append(Link_after_loop)
            R_after=R_after*(1-alpha[position]*gamma[position][0])
#            R_after=R_after*(1-alpha[position])
            R_after=tf.concat( [R_after, [1.0]],0)
    
    forget_index_TEST.append(forget_index)
    forgetcoef_TEST.append(forgetcoef)
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
Num_max=60
#Centerincrease=np.zeros((Num_max))
Center=np.zeros((Num_max,Dimx))
Center_weight=np.zeros((Num_max,Dim))   
print('1021')
L_curiculum=20
Error_ave=0
for i in range (N_tr):
#    l1=np.random.randint(3)+2
#    l=2*l1
#    l=np.random.randint(10)+2
    [data_onehot,X,T,M1,Leffect]=getTrainBatch(i%2000,L_curiculum)
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
    X_dic_current_iterate=np.asarray(X_shink)
#    gamma_v,gammaxxx_v,R_after_v=sess.run([gamma,gammaxxx,R_after],{data: data_onehot, label: T, y:y1,mask:MASK, M:MM,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})

    forgetcoef_TEST_v,forget_index_TEST_v,y2_TEST_v,R_after_TEST_v,Readweight_loop_TEST_v,Readweight123_TEST_v,Link_after_loop_TEST_v,gammaxxx_v,\
    Link_after_loop_v,Link_after_norm2_v,gamma_v,Link_pre_NORM_TEST_v,Link_pre_loop_TEST_v,\
    alpha_v,optimizer_v,error_v,pred_V,loss_v=\
    sess.run([forgetcoef_TEST,forget_index_TEST,y2_TEST,R_after_TEST,Readweight_loop_TEST,Readweight123_TEST,Link_after_loop_TEST,gammaxxx,\
    Link_after_loop,Link_after_norm2,gamma,Link_pre_NORM_TEST,Link_pre_loop_TEST,\
    alpha_final,optimizer,error,Pred_final,loss], \
    {data: data_onehot, label: T, y:y1,mask:M1, M:X_dic_current_iterate,inputdata:X,Linkage:Linkage_ini,Readweight:Readweight_ini})
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
    Error_ave=Error_ave+mse_tr[i]
    if (i % 100 == 0):
        print (mse_tr[i],Error_ave,L_curiculum)     
        
        if (Error_ave<5e-2):
            L_curiculum=L_curiculum+1
#            print (mse_tr[i],i)
#            print (alpha_v,gamma_v)
#            inverted_label = label_encoder.inverse_transform(np.argmax(T, axis=1))
#            inverted_pred = label_encoder.inverse_transform(np.argmax(pred_V, axis=1))
#            print('label',inverted_label)
#            print('pred',inverted_pred)
        Error_ave=0
            
            
#        print (alpha_v,gamma_v)
#        inverted_label = label_encoder.inverse_transform(np.argmax(T, axis=1))
#        inverted_pred = label_encoder.inverse_transform(np.argmax(pred_V, axis=1))
#        print('label',inverted_label)
#        print('pred',inverted_pred)
    if (i % 2000==0):
        print (mse_tr[i],i)
#        print (alpha_v,gamma_v)
        inverted_label = label_encoder.inverse_transform(np.argmax(T, axis=1))
        inverted_pred = label_encoder.inverse_transform(np.argmax(pred_V, axis=1))
#        print('label',inverted_label)
#        print('pred',inverted_pred)
#        print ('predict',PRED,'target',T)
#        print (alpha_v,gamma_v)
#    if ((i>10000)&(mse_tr[i]>0.01)):
#        print ('predict',PRED,'target',T,mse_tr[i],i)
