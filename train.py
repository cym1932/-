##from tensorflow.contrib import learn
import tensorflow as tf
import data_helper
import numpy as np
import os
import sys
import json
import time
import math
import parameters

def shuffle_batch(user,item,score, batch_size):
    rnd_idx = np.random.permutation(len(score))
    n_batches = len(score) // batch_size
    rnd_idx = rnd_idx[0:n_batches*batch_size]
    for batch_idx in np.array_split(rnd_idx, n_batches):
        user_batch = user[:,batch_idx,:]
        item_batch = item[:,batch_idx,:]
        y_batch = score[batch_idx]
        yield user_batch,item_batch,y_batch

    
def train(user=None,item=None,score=None,
          user_dev=None,item_dev=None,score_dev=None,
          n_epochs=1000, batch_size=20,learn_rate=1e-3,evaluate_every=15):

        graph=tf.Graph()
        k=parameters.dimension
        n_graph=parameters.n_graph
        

        with graph.as_default():
                sess=tf.Session()
                with sess.as_default():
                        print(1)
                        y_train=tf.placeholder(
                                tf.float64,shape=(None,),name='y_train')
##                        y_train_one_hot=tf.placeholder(
##                                tf.float64,shape=(None,5),name='y_train')
                        user_train=tf.placeholder(
                                tf.float64,shape=(n_graph,None,k),name='user_train')
                        item_train=tf.placeholder(
                                tf.float64,shape=(n_graph,None,k),name='item_train')

##                        with tf.name_scope('concatenate'):
##                            user_final=tf.reshape(tf.transpose(user_train,[1,0,2]),
##                                                  [-1,n_graph*k])
##                            item_final=tf.reshape(tf.transpose(item_train,[1,0,2]),
##                                                  [-1,n_graph*k])
                            
                        with tf.name_scope('attention'):
                                att_w1=tf.Variable(
                                        np.random.normal(-2,2,[k,k]),dtype=tf.float64)
                                att_w1_final=tf.stack([att_w1]*n_graph)
                                att_b1=tf.Variable(
                                        np.random.uniform(-2,2,[k]),dtype=tf.float64)

                                
                                u_attention_1=tf.nn.relu(
                                        tf.matmul(user_train,att_w1_final)+att_b1)
                                i_attention_1=tf.nn.relu(
                                        tf.matmul(item_train,att_w1_final)+att_b1)

                                att_w2=tf.Variable(
                                        np.random.normal(-2,2,[k,1]),dtype=tf.float64)
                                att_w2_final=tf.stack([att_w2]*n_graph)
                                u_attention_weights=tf.nn.softmax(
                                        tf.reshape(tf.transpose(tf.matmul(
                                        u_attention_1,att_w2_final),[1,2,0])
                                        ,[-1,n_graph]))
                                i_attention_weights=tf.nn.softmax(
                                        tf.reshape(tf.transpose(tf.matmul(
                                        i_attention_1,att_w2_final),[1,2,0])
                                        ,[-1,n_graph]))
                                
                                user_transposed=tf.transpose(user_train,[2,1,0])
                                item_transposed=tf.transpose(item_train,[2,1,0])
                                user_final=tf.reduce_sum(tf.transpose(
                                        user_transposed*u_attention_weights,[2,1,0]),axis=[0])
                                item_final=tf.reduce_sum(tf.transpose(
                                        item_transposed*i_attention_weights,[2,1,0]),axis=[0])

##                        user_final=tf.reduce_mean(user_train,0)
##                        item_final=tf.reduce_mean(item_train,0)

##                        user_final=user_train[0]
##                        item_final=item_train[0]
                        
                        x_train=tf.concat([user_final,item_final],1)
##                        x_train=user_final*item_final
                        
                        with tf.name_scope("dnn"):
                                hidden1 = tf.layers.dense(x_train,
                                                          parameters.h1_width,
                                                          name="hidden1",
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.truncated_normal_initializer(\
                                                              mean=0,stddev=.2),
                                                          bias_initializer=tf.truncated_normal_initializer(\
                                                              mean=.2,stddev=.1)
                                                          )
                                hidden2 = tf.layers.dense(hidden1,
                                                          parameters.h2_width,
                                                          name="hidden2",
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.truncated_normal_initializer(\
                                                              mean=0,stddev=.15),
                                                          bias_initializer=tf.truncated_normal_initializer(\
                                                              mean=-.1,stddev=.1)
                                                          )
    
                                result = tf.layers.dense(hidden2, 1, name="result")

                                
                        result=tf.reshape(result,[-1])

                        diff=y_train-result
                        square=tf.multiply(diff,diff)
##                        mean=tf.reduce_mean(result)
##                        mean2=tf.reduce_mean(y_train)
                        
                       #loss=tf.norm(result,2)
                        loss=tf.sqrt(tf.reduce_mean(square))

                        train_step=tf.train.AdamOptimizer(learning_rate=learn_rate).\
                                     minimize(loss)
                        
                        init = tf.global_variables_initializer()
                        sess.run(init)
##                        print(sess.run(result,feed_dict=
##                                       {user_train:user,item_train:item,y_train:score})[0:12])

                        #start training#
                        best_loss=1e99
                        best_epoch=-1

                        best_result=None
                        current_time=str(time.time())
                        print(current_time)
                        for epoch in range(n_epochs):
                                if epoch-best_epoch>50:
                                        break
                                for u_batch,i_batch,y_batch in \
                                    shuffle_batch(user,item,score,batch_size):
##                                        one_hot=np.eye(5)[score+3]
                                        feed_dict={user_train:u_batch,
                                                   item_train:i_batch,
                                                   y_train:y_batch}
                                        sess.run(train_step,feed_dict)
                                if epoch%evaluate_every==0 or epoch<evaluate_every:
                                        feed_dict={user_train:user,
                                                   item_train:item,
                                                   y_train:score}
##                                        print(sess.run(result,feed_dict))

                                        dev_loss=sess.run(loss,feed_dict=
                                                          {user_train:user_dev,
                                                           item_train:item_dev,
                                                           y_train:score_dev})
                                        if dev_loss<best_loss:
                                                best_loss=dev_loss
                                                best_epoch=epoch
                                                best_result=sess.run(result,feed_dict=
                                                          {user_train:user_dev,
                                                           item_train:item_dev,
                                                           y_train:score_dev})
##                                                save_path=tf.train.Saver().save(sess,
##                                                                     './models'+current_time+'/'+str(epoch)+'.ckpt')
                                        print('epoch=',epoch,'\t',
                                              'train=%.4f'%sess.run(loss,feed_dict),'\t',
                                              'dev=%.4f'%dev_loss,'\t',
                                              'best=%.4f'%best_loss,' @ ',best_epoch)
                                    
        return best_result
    
def read_embedding():
        folder_path=parameters.folder_name
        filenames=['1','2','3','4','5','6','7','8']
        filenames=filenames[0:parameters.n_graph]
        k=parameters.dimension
        user_full=[]
        item_full=[]
        for filename in filenames:
                with open(folder_path+str(parameters.embedding_method)+filename+'.txt','r',encoding='utf8') as fp:
                        n_users=int(fp.readline())
                        user_g=[]
                        for i in range(n_users):
                                line=fp.readline()
                                a=[float(i) for i in line.split()]
                                a=a[0:k]
                                user_g.append(a)
                        n_items=int(fp.readline())
                        item_g=[]
                        for i in range(n_items):
                                line=fp.readline()
                                a=[float(i) for i in line.split()]
                                a=a[0:k]
                                item_g.append(a)
                        user_full.append(user_g)
                        item_full.append(item_g)
        user_embedding=np.array(user_full)
        item_embedding=np.array(item_full)
        #print(user_embedding.shape)
        if parameters.dataset=='yelp':
                train_set_filename='user_business_short.txt'
                dev_set_filename='user_business_short_test.txt'
        else:
                train_set_filename='a_user_item_short.txt'
                dev_set_filename='a_user_item_short_test.txt'
                
                
        k=parameters.dimension
        n_graph=parameters.n_graph
        
        with open(folder_path+train_set_filename,'r',encoding='utf8') as fp:
                lines=fp.readlines()
                user_train=np.zeros([n_graph,len(lines)-1,k])
                item_train=np.zeros([n_graph,len(lines)-1,k])
                score_train=np.zeros(len(lines)-1)
                for j in range(len(lines)):
                        if j>0:
                                line=lines[j]
                                a=[int(float(i)) for i in line.split()]
                                user_train[:,j-1,:]=user_embedding[:,a[0],:]
                                item_train[:,j-1,:]=item_embedding[:,a[1],:]
                                score_train[j-1]=a[2]

        with open(folder_path+dev_set_filename,'r',encoding='utf8') as fp:
                lines=fp.readlines()
                user_dev=np.zeros([n_graph,len(lines)-1,k])
                item_dev=np.zeros([n_graph,len(lines)-1,k])
                score_dev=np.zeros(len(lines)-1)
                for j in range(len(lines)):
                        if j>0:
                                line=lines[j]
                                a=[int(float(i)) for i in line.split()]
                                user_dev[:,j-1,:]=user_embedding[:,a[0],:]
                                item_dev[:,j-1,:]=item_embedding[:,a[1],:]
                                score_dev[j-1]=a[2]

        
        return user_train,item_train,score_train,user_dev,item_dev,score_dev
        
if __name__ == '__main__':
        user_train,item_train,score_train,user_dev,item_dev,score_dev = read_embedding()

        best=train(user=user_train,item=item_train,score=score_train,
              user_dev=user_dev,item_dev=item_dev,score_dev=score_dev,
              n_epochs=2000,batch_size=parameters.batch_size,
              evaluate_every=5,learn_rate=parameters.learn_rate)
        with open('./best.txt','w') as b:
                for i in range(best.shape[0]):
                        b.write(str(best[i]))
                        b.write('\n')

        print(score_dev)
