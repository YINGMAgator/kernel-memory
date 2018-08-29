import numpy as np
from tomita import trainlab,strain,testlab,stest,mask,maskv
import tensorflow as tf
# rnn model
batchSize = 1
iterations = 1000
maxSeqLength = 15
numDimensions = 1 #Dimensions for each word vector
hiddendimension=4
epoch_num=100

tf.reset_default_graph()
label= tf.placeholder(tf.float32, [batchSize, maxSeqLength,numDimensions])
data = tf.placeholder(tf.float32, [batchSize, maxSeqLength,numDimensions])
inputmask = tf.placeholder(tf.int32, [batchSize, maxSeqLength,1])

# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
rnncell = tf.contrib.rnn.BasicRNNCell(hiddendimension)
value, value1 = tf.nn.dynamic_rnn(rnncell, data, dtype=tf.float32)
#layer1
weight = tf.Variable(tf.truncated_normal([hiddendimension, hiddendimension]))
bias = tf.Variable(tf.constant(0.1, shape=[hiddendimension]))
# value = tf.transpose(value, [1, 0, 2])
l_out_x = tf.reshape(value, [-1, hiddendimension], name='2_2D')
# shape = (batch * steps, output_size)
pred1 = tf.matmul(l_out_x, weight) + bias
#layer2
weight1 = tf.Variable(tf.truncated_normal([hiddendimension, numDimensions]))
bias1 = tf.Variable(tf.constant(0.1, shape=[numDimensions]))
# value = tf.transpose(value, [1, 0, 2])
# l_out_x = tf.reshape(value, [-1, lstmUnits], name='2_2D')
# shape = (batch * steps, output_size)
pred = tf.matmul(pred1, weight1) + bias1
#
labels=tf.reshape(label, [-1, numDimensions])
maskaa=tf.reshape(inputmask, [-1, 1])
loss=tf.losses.mean_squared_error(
    labels,
    pred,
    weights=maskaa,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
)
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range (epoch_num):
    for i in range(iterations):
        inputdata =np.expand_dims(strain[i], axis=0)
        inputlabel=np.expand_dims(trainlab[i], axis=0)
        trainmask=np.expand_dims(mask[i], axis=0)
        # valuev,value1v,maskav,lossv=sess.run([pred,labels,maskaa,loss], {data: inputdata, label: inputlabel, inputmask:trainmask})
        sess.run(optimizer, {data: inputdata, label: inputlabel, inputmask:trainmask})

        # print(valuev,value1v,maskav,lossv)
        if (i % 100 == 0):
            error_ave=0
            for j in range (np.shape(stest)[0]):
                inputdata =np.expand_dims(stest[j], axis=0)
                inputlabel=np.expand_dims(testlab[j], axis=0)
                testmask=np.expand_dims(maskv[j], axis=0)
                ls= sess.run(loss, {data: inputdata, label: inputlabel, inputmask:testmask})
                error_ave=error_ave+ls
            print(error_ave/np.shape(stest)[0])
        if (i % 999 == 0):
            j=np.random.randint(np.shape(stest)[0])
            inputdata =np.expand_dims(stest[j], axis=0)
            inputlabel=np.expand_dims(testlab[j], axis=0)
            testmask=np.expand_dims(maskv[j], axis=0)
            ls,pd,labelsv,maskaav = sess.run([loss,pred,labels,maskaa], {data: inputdata, label: inputlabel, inputmask:testmask})
            print (ls,pd,labelsv,maskaav)
