from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf
import math
import funksvd
import parameters

def row_normalization(mat):
        indptr=mat.indptr
        indices=mat.indices 
        data=mat.data

        row=mat.shape[0]
        col=mat.shape[1]
        x_index=0
        
        for i in range(indices.shape[0]):
                y_index=indices[i]
                while i>=indptr[x_index+1]:
                        x_index+=1
                data[i]=1.0/(indptr[x_index+1]-indptr[x_index])
        return csr_matrix((data,indices,indptr),shape=mat.shape)

if __name__ == '__main__':
        folder_name=parameters.folder_name
        filenames=['business_category',
                   'business_city',
                   'business_star',
                   'business_state',
                   'review_business',
                   'review_user',
                   'user_business',
                   'user_user']
        suffix='_short.txt'
        dict_mat={}

        item_count=0
        user_count=0
        global_mean=0
        global_count=0
        statistics=np.zeros(7)
        with open(folder_name+'user_business'+suffix,'r',encoding='utf8') as file:
                line=file.readline()
                user_count=int(line.split()[0])
                item_count=int(line.split()[1])
                while True:
                        line=file.readline()
                        if not line:
                                break
                        a=[float(i) for i in line.split()]
                        global_mean+=a[2]
                        global_count+=1.0
                        statistics[int(a[2])]+=1
        global_mean=global_mean/global_count
        print(global_mean)
        print(statistics)
        score_transform=np.zeros(6)
        score_transform[0]=0.5
        for i in range(1,6):
                score_transform[i]=score_transform[i-1]+\
                        (statistics[i-1]+statistics[i])/global_count*2.5
        print(score_transform)

        for filename in filenames: 
                fname=folder_name+filename+suffix
                row=[]
                col=[]
                data=[]
                with open(fname,'r',encoding='utf8') as file:
                        line=file.readline()
                        a=[int(i) for i in line.split()]
                        maxi=a[0]
                        maxj=a[1]
                        while True:
                                line=file.readline()
                                if not line:
                                        break
                                a=[int(float(i)) for i in line.split()]
                                if maxi<a[0]+1:
                                        maxi=a[0]+1
                                if maxj<a[1]+1:
                                        maxj=a[1]+1
                                row.append(a[0])
                                col.append(a[1])
                                if filename=='user_business':
##                                        data.append(a[2])
                                        data.append(score_transform[int(a[2])])
                                else:
                                        data.append(1)
                                        if filename=='business_star':
                                                if a[1]<8:
                                                        row.append(a[0])
                                                        col.append(a[1]+1)
                                                        data.append(parameters.embedding_star_adjacency)
                                                if a[1]>0:
                                                        row.append(a[0])
                                                        col.append(a[1]-1)
                                                        data.append(parameters.embedding_star_adjacency)
                                                        
                                #data.append(1)
                                
                mat=csr_matrix((data,(row,col)),shape=(maxi,maxj),dtype=float)
##                mat=row_normalization(mat)
                mat2=csr_matrix((data,(col,row)),shape=(maxj,maxi),dtype=float)
##                mat2=row_normalization(mat2)
                dict_mat[filename]=mat
                dict_mat[filename+'_invert']=mat2


        checkin=dict_mat['user_business'].copy()
        for i in range(checkin.data.shape[0]):
                checkin.data[i]=1
        dict_mat['checkin']=checkin

        average_rating=np.zeros(item_count)
        for i in range(item_count):
                total=0
                count=0
                for j in range(user_count):
                        rating=dict_mat['user_business'][j,i]
                        if rating!=0:
                                total+=rating
                                count+=1.0
                if count>0:
                        average_rating[i]=total/count
                else:
                        average_rating[i]=3.0

        for i in range(dict_mat['user_business'].data.shape[0]):
                dict_mat['user_business'].data[i]-=\
                                                     3
##                        average_rating[dict_mat['user_business'].indices[i]]-(1e-12)
        dict_mat['user_business_invert']=np.transpose(dict_mat['user_business'])


        
        mats=[]
        #UB
        mats.append(dict_mat['user_business'])
        #UUB
        mats.append(dict_mat['user_user']*dict_mat['user_business'])
        #UBUB
        mats.append(dict_mat['user_business']*dict_mat['user_business_invert']   \
              *dict_mat['user_business'])
        #UBCB
        mats.append(dict_mat['user_business']*dict_mat['business_category']      \
              *dict_mat['business_category_invert'])
        #UBCBUB
        mats.append(dict_mat['user_business']*dict_mat['business_category']      \
              *dict_mat['business_category_invert']*dict_mat['user_business_invert']   \
              *dict_mat['user_business'])
        #UBCB
        mats.append(dict_mat['user_business']*dict_mat['business_city']          \
              *dict_mat['business_city_invert'])

        #UBCBUB
        mats.append(dict_mat['user_business']*dict_mat['business_city']      \
              *dict_mat['business_city_invert']*dict_mat['user_business_invert']   \
              *dict_mat['user_business'])

        #UBSB
        mats.append(dict_mat['user_business']*dict_mat['business_star']          \
              *dict_mat['business_star_invert'])

        for matrix in mats:
                count=0
                count2=0
                sum=0
                sum2=0

                for i in range(matrix.data.shape[0]):
                        if matrix.data[i]!=0:
                                count2+=1.0

##                                ##matrix data change
                                q=matrix.data[i]
                                matrix.data[i]=np.sign(q)*math.log(1+abs(q))
                                sum+=abs(matrix.data[i])
                                sum2+=pow(matrix.data[i],2)
                                
                print(sum/count2,sum2/count2)

        n_graph=parameters.n_graph
        mats_np=np.zeros([n_graph,user_count,item_count])
        om_np=np.zeros([n_graph,user_count,item_count])
        for i in range(n_graph):
                mats_np[i]=mats[i].toarray()
                om_np[i]=checkin.toarray()

        print(mats_np[0])
        print(om_np[0])
##        mats.append(dict_mat['checkin']*dict_mat['business_state']         \
##              *dict_mat['business_state_invert'])


##        matrices_indicated=[]
##        for matrix in mats:
##                indptr=om.indptr
##                indices=om.indices
##                data=om.data.copy()
##                
##                row=om.shape[0]
##                col=om.shape[1]
##                x_index=0
##
##                for i in range(indices.shape[0]):
##                        y_index=indices[i]
##                        while i>=indptr[x_index+1]:
##                                x_index+=1
##                        data[i]=matrix[x_index,y_index]
##
##                temp_matrix=csr_matrix((data,indices,indptr),shape=om.shape,dtype=float)        
##                matrices_indicated.append(temp_matrix)

        print('===') 
        num=0

        graph=tf.Graph()
        row=user_count
        col=item_count
        count=0
        best_loss=1e99
        bestx=0
        besty=0
        k=parameters.dimension
        with graph.as_default():
                sess=tf.Session()
                with sess.as_default():
                        param_lambda=tf.constant(parameters.l2_norm_pen,dtype=tf.float64)
                        target=tf.constant(mats_np,dtype=tf.float64)
                        target_=tf.reshape(target,[-1,col])
                        om=tf.constant(om_np,dtype=tf.float64)
                        om_=tf.reshape(om,[-1,col])
                        x=tf.Variable(tf.random_uniform((n_graph,row,k),minval=-1,maxval=1,dtype=tf.float64))
                        x_=tf.reshape(x,[-1,k])

                        y=tf.Variable(tf.random_uniform((col,k),minval=-1,maxval=1,dtype=tf.float64))
                        x_mul_y=tf.matmul(x_,y,transpose_b=True)
                        error=tf.multiply(om_,tf.subtract(x_mul_y,target_))
                        loss=tf.add(tf.pow(tf.norm(error,ord=2),2),
                        tf.multiply(param_lambda,tf.add(tf.pow(tf.norm(x,ord=2),2),
                                                        tf.pow(tf.norm(y,ord=2),2)*n_graph)))
                        
                        train_step=tf.train.AdamOptimizer(parameters.learn_rate_embedding).\
                                    minimize(loss)

                        init = tf.global_variables_initializer()
                        sess.run(init)
                        print()
                        print(sess.run(loss))
                       
                        loss_old=sess.run(loss)
                        n=0
                        while True:
                                sess.run(train_step)
##                                print(1)
                                n+=1
                                if n%100==0:
                                        loss_new=sess.run(loss)
                                        print(n,loss_new)
                                        if loss_new<best_loss:
                                                best_loss=loss_new
                                                bestx=sess.run(x)
                                                besty=sess.run(y)
                                        if loss_new-loss_old>-0.02:
                                                count+=1
                                                if count>3:
                                                        break
                                        else:
                                                count-=2
                                                if count<0:
                                                        count=0
                                        loss_old=loss_new
                        x=sess.run(x)
                        y=sess.run(y)

                        for num in range(1,1+n_graph):
                                with open(folder_name+'e'+str(num)+'.txt','w',encoding='utf8') as fp:
                                        fp.write(str(row)+'\n')
                                        for i in range(row):
                                                for j in range(k):
                                                        fp.write(str(x[num-1,i,j])+'\t')
                                                fp.write('\n')
                                        fp.write(str(col)+'\n')
                                        for i in range(col):
                                                for j in range(k):
                                                        fp.write(str(y[i,j])+'\t')
                                                fp.write('\n')
                                
                 
