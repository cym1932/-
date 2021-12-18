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

def make_sim(dic, files, user_count):
        ret=np.zeros([user_count,user_count])
        b=csr_matrix(([0],([0],[0])),shape=(user_count,user_count),dtype=float)
        for i in range(1,6):
                a=1
                for file in files:
                        if file=='a_user_item' or file=='user_business'\
                           or file=='a_user_item_invert' or file=='user_business_invert':
                                file=file+str(i)
                        a=a*dic[file]
                b=b+a

        b=b.toarray()
        for i in range(user_count):
                q=b[i,i]
                for j in range(user_count):
                        if i!=j:
                                if q>0:
                                        ret[i,j]=b[i,j]/q
                                else:
                                        ret[i,j]=0


        for i in range(user_count):
                ret[i,i]=1                                
        return ret

def non_0_count(x):
        s=0
        for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                        if x[i,j]!=0:
                                s+=1
        return s

if __name__ == '__main__':
        emmax=[0,1,1,1,1,1,2,1]
        for i in range(1):
                print(parameters.embedding_method)
                
                folder_name=parameters.folder_name
                if parameters.dataset=='yelp':
                        filenames=['user_business',
                                   'business_category',
                                   'business_city',
                                   'business_star',
                                   'business_state',
                                   'review_business',
                                   'review_user',
                                   'user_user']
                else:
                        filenames=['a_user_item',
                                   'a_item_brand',
                                   'a_item_category']
                suffix='_short.txt'
                dict_mat={}

                item_count=0
                user_count=0
                global_mean=0
                global_count=0
                statistics=np.zeros(7)
                with open(folder_name+filenames[0]+suffix,'r',encoding='utf8') as file:
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

                aaaa=input()
                print(aaaa)
                input()

                sqerror_original=0.0
                for i in range(1,6):
                        sqerror_original+=(i-global_mean)*(i-global_mean)*statistics[i]
                sqerror_original=sqerror_original/global_count
                print(sqerror_original)
                
                sqerror_modified=0.0
                for i in range(1,6):
                        sqerror_modified+=(score_transform[i]-3)*(score_transform[i]-3)*statistics[i]
                sqerror_modified=sqerror_modified/global_count
                print(sqerror_modified)
                
                print(sqerror_original,sqerror_modified)
                for i in range(1,6):
                        q=score_transform[i]
                        score_transform[i]=(q-3)*pow(sqerror_original/sqerror_modified,0.5)+3
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
                                        if filename=='user_business' or\
                                           filename=='a_user_item':
        ##                                        data.append(a[2])
                                                if parameters.embedding_method[6]==2:
                                                        data.append(score_transform[int(a[2])])
                                                elif parameters.embedding_method[6]==0:
                                                        data.append(1)
                                                else:
                                                        data.append(a[2])
                                        else:
                                                data.append(1)
                                                if filename=='business_star' and parameters.embedding_method[7]==1:
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
                        
                        mat2=csr_matrix((data,(col,row)),shape=(maxj,maxi),dtype=float)
                        if parameters.embedding_method[4]==1 and filename!='user_business':
                                mat=row_normalization(mat)
                                mat2=row_normalization(mat2)
                        dict_mat[filename]=mat
                        dict_mat[filename+'_invert']=mat2

                if parameters.dataset=='yelp':
                        checkin=dict_mat['user_business'].copy()
                else:
                        checkin=dict_mat['a_user_item'].copy()
                for i in range(checkin.data.shape[0]):
                        checkin.data[i]=1
                dict_mat['checkin']=checkin

                average_rating=np.zeros(item_count)
##                for i in range(item_count):
##                        total=0
##                        count=0
##                        for j in range(user_count):
##                                rating=dict_mat['user_business'][j,i]
##                                if rating!=0:
##                                        total+=rating
##                                        count+=1.0
##                        if count>0:
##                                average_rating[i]=total/count
##                        else:
##                                average_rating[i]=3.0
                if parameters.dataset=='yelp':
                        for i in range(dict_mat['user_business'].data.shape[0]):
                                if parameters.embedding_method[2]==1:
                                        dict_mat['user_business'].data[i]-=\
                                                                             3
                                if parameters.embedding_method[3]==1:
                                        dict_mat['user_business'].data[i]-=\
                                                                             average_rating[dict_mat['a_user_item'].indices[i]]-(1e-12)
                        dict_mat['user_business_invert']=np.transpose(dict_mat['user_business'])
                else:
                        for i in range(dict_mat['a_user_item'].data.shape[0]):
                                if parameters.embedding_method[2]==1:
                                        dict_mat['a_user_item'].data[i]-=\
                                                                             3
                                if parameters.embedding_method[3]==1:
                                        dict_mat['a_user_item'].data[i]-=\
                                                                             average_rating[dict_mat['a_user_item'].indices[i]]-(1e-12)
                        dict_mat['a_user_item_invert']=np.transpose(dict_mat['a_user_item'])


                if not parameters.semrec:
                        mats=[]
                        if parameters.dataset=='yelp':
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
                                
                                #UBCB
                                mats.append(dict_mat['user_business']*dict_mat['business_city']          \
                                      *dict_mat['business_city_invert'])
                                
                                #UBSB
                                mats.append(dict_mat['user_business']*dict_mat['business_star']          \
                                      *dict_mat['business_star_invert'])
                                
                                #UBCBUB
                                mats.append(dict_mat['user_business']*dict_mat['business_category']      \
                                      *dict_mat['business_category_invert']*dict_mat['user_business_invert']   \
                                      *dict_mat['user_business'])
                                
                                #UBCBUB
                                mats.append(dict_mat['user_business']*dict_mat['business_city']      \
                                      *dict_mat['business_city_invert']*dict_mat['user_business_invert']   \
                                      *dict_mat['user_business'])
                        else:
                                #UB
                                mats.append(dict_mat['a_user_item'])
                                
                                #UBUB
                                mats.append(dict_mat['a_user_item']*dict_mat['a_user_item_invert']   \
                                      *dict_mat['a_user_item'])
                                
                                #UBCB
                                mats.append(dict_mat['a_user_item']*dict_mat['a_item_category']      \
                                      *dict_mat['a_item_category_invert'])
                                
                                #UBbB
                                mats.append(dict_mat['a_user_item']*dict_mat['a_item_brand']      \
                                      *dict_mat['a_item_brand_invert'])
                                
                                #UBCBUB
                                mats.append(dict_mat['a_user_item']*dict_mat['a_item_category']      \
                                      *dict_mat['a_item_category_invert']*dict_mat['a_user_item_invert']   \
                                      *dict_mat['a_user_item'])
                                
                                #UBbBUB
                                mats.append(dict_mat['a_user_item']*dict_mat['a_item_brand']      \
                                      *dict_mat['a_item_brand_invert']*dict_mat['a_user_item_invert']   \
                                      *dict_mat['a_user_item'])
                              

                        
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
                        need_to_update=[7,8]
                        for matrix in mats:
                                num+=1
                                print(num)
                                om=mats[0].copy()
                                if parameters.embedding_method[1]==1:
                                        om=matrix.copy()
                                count=0
                                count2=0
                                sum=0
                                sum2=0
                                for i in range(om.data.shape[0]):
                                        if om.data[i]!=0:
                                                om.data[i]=1
                                                count+=1.0

                                for i in range(matrix.data.shape[0]):
                                        if matrix.data[i]!=0:
                                                count2+=1.0

                ##                                ##matrix data change
                                                if parameters.embedding_method[5]==1:
                                                        q=matrix.data[i]
                                                        matrix.data[i]=np.sign(q)*math.log(1+abs(q))

                ##                                print(matrix.data[i])
                                                sum+=abs(matrix.data[i])
                                                sum2+=pow(matrix.data[i],2)
                                                
                                print(count,sum/count2,sum2/count2)

                                ############
                ##                om=mats[0]
                                ############
                                
                                if num in need_to_update:
                                        k=parameters.dimension
                                        x,y,c,n=funksvd.funksvd_tf(matrix,om,k=k,alpha=parameters.learn_rate_embedding,
                                                                   lamda=parameters.l2_norm_pen,
                                                                   evaluate_every=100)
                                        with open(folder_name+str(parameters.embedding_method)+str(num)+'.txt','w',encoding='utf8') as fp:
                                                row=x.shape[0]
                                                col=y.shape[0]
                                                fp.write(str(row)+'\n')
                                                for i in range(row):
                                                        for j in range(k):
                                                                fp.write(str(x[i,j])+'\t')
                                                        fp.write('\n')
                                                fp.write(str(col)+'\n')
                                                for i in range(col):
                                                        for j in range(k):
                                                                fp.write(str(y[i,j])+'\t')
                                                        fp.write('\n')
                                                
                        flag=True
                        j=7
                        while parameters.embedding_method[j]>=emmax[j] and j>0:
                                j=j-1
                        if True:
                                break
                        else:
                                parameters.embedding_method[j]+=1
                                for i in range(j+1,8):
                                        parameters.embedding_method[i]=0
                         
                else:
                        #######semrec#####
                        print('s')
                        user_count=-1
                        for filename in filenames: 
                                fname=folder_name+filename+suffix
                                if filename!='user_business' and filename!='a_user_item':
                                        row=[]
                                        col=[]
                                        data=[]
                                else:
                                        row=[[],[],[],[],[],[]]
                                        col=[[],[],[],[],[],[]]
                                        data=[[],[],[],[],[],[]]
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
                                                if filename=='user_business' or\
                                                   filename=='a_user_item':
                                                        row[a[2]].append(a[0])
                                                        col[a[2]].append(a[1])
                                                        data[a[2]].append(1)
                                                        row[0].append(a[0])
                                                        col[0].append(a[1])
                                                        data[0].append(a[2])
                                                else:
                                                        row.append(a[0])
                                                        col.append(a[1])
                                                        data.append(1)
                                                                     
                                if filename!='user_business' and filename!='a_user_item':   
                                        mat=csr_matrix((data,(row,col)),shape=(maxi,maxj),dtype=float)
                                        mat2=csr_matrix((data,(col,row)),shape=(maxj,maxi),dtype=float)
                                        dict_mat[filename]=mat
                                        dict_mat[filename+'_invert']=mat2
                                else:
                                        for i in range(6):
                                                mat=csr_matrix((data[i],(row[i],col[i])),shape=(maxi,maxj),dtype=float)
                                                mat2=csr_matrix((data[i],(col[i],row[i])),shape=(maxj,maxi),dtype=float)
                                                dict_mat[filename+str(i)]=mat
                                                dict_mat[filename+'_invert'+str(i)]=mat2

##                        print(2)
                        if parameters.dataset=='amazon':
                                paths=[]
                                user_count=dict_mat['a_user_item1'].shape[0]
                                item_count=dict_mat['a_user_item1'].shape[1]
                                paths.append(['a_user_item','a_user_item_invert'])
                                paths.append(['a_user_item','a_item_brand','a_item_brand_invert','a_user_item_invert'])
                                paths.append(['a_user_item','a_item_category','a_item_category_invert','a_user_item_invert'])
                                rate=dict_mat['a_user_item0']
                        else:
                                paths=[]
                                user_count=dict_mat['user_business1'].shape[0]
                                item_count=dict_mat['user_business1'].shape[1]
                                paths.append(['user_business','user_business_invert'])
                                paths.append(['user_business','business_category','business_category_invert','user_business_invert'])
                                paths.append(['user_business','business_city','business_city_invert','user_business_invert'])
                                paths.append(['user_business','business_star','business_star_invert','user_business_invert'])
                                rate=dict_mat['user_business0']
                        sim=np.zeros([len(paths),user_count,user_count])
                        predict=np.zeros([len(paths),user_count,item_count])
                        count=np.zeros([len(paths),user_count,item_count])

##                        print(3)
                        rate=rate.toarray()
                        for i in range(len(paths)):
                                print(i)
                                sim[i]=make_sim(dict_mat,paths[i],user_count)
##                                print(sim[i])
                                for k in range(user_count): 
##                                        print(k)
                                        for y in range(item_count):
                                                if rate[k,y]>0:
                                                        for x in range(user_count):
                                                                if x!=k:
                                                                        predict[i,x,y]+=rate[k,y]*sim[i,x,k]
                                                                        count[i,x,y]+=sim[i,x,k]

                        for i in range(len(paths)):
                                for x in range(user_count):
                                        for y in range(item_count):
                                                if count[i,x,y]>0:
                                                        predict[i,x,y]=predict[i,x,y]/count[i,x,y]
##                                                        if i>=1:
##                                                                print(predict[i,x,y])
                                                else:
                                                        predict[i,x,y]=1
                                                        
                                                
                        if parameters.dataset=='yelp':
                                train_set_filename='user_business_short.txt'
                                dev_set_filename='user_business_short_test.txt'
                        else:
                                train_set_filename='a_user_item_short.txt'
                                dev_set_filename='a_user_item_short_test.txt'
                        
                        with open(folder_name+train_set_filename,'r',encoding='utf8') as fp:
                                lines=fp.readlines()
                                score_train=np.zeros([user_count,item_count])
                                mask_train=np.zeros([user_count,item_count])
                                for j in range(len(lines)):
                                        if j>0:
                                                line=lines[j]
                                                a=[int(float(i)) for i in line.split()]
                                                score_train[a[0],a[1]]=a[2]
                                                mask_train[a[0],a[1]]=1

                        with open(folder_name+dev_set_filename,'r',encoding='utf8') as fp:
                                lines=fp.readlines()
                                score_dev=np.zeros([user_count,item_count])
                                mask_dev=np.zeros([user_count,item_count])
                                for j in range(len(lines)):
                                        if j>0:
                                                line=lines[j]
                                                a=[int(float(i)) for i in line.split()]
                                                score_dev[a[0],a[1]]=a[2]
                                                mask_dev[a[0],a[1]]=1
                                                
                        graph=tf.Graph()
                        

                        with graph.as_default():
                                sess=tf.Session()
                                with sess.as_default():
                                        print(1)
                                        w=np.random.uniform(0.2,0.4,[len(paths)])
                                        w[0]-=(np.sum(w)-1)
                                        rui=tf.constant(predict,name='rui')
                                        x_train=tf.placeholder(tf.float64,shape=(user_count,item_count),
                                                               name='x_train')
                                        x_mask=tf.placeholder(tf.float64,shape=(user_count,item_count),
                                                               name='x_mask')
                                       
                                        weights=tf.Variable(w,dtype=tf.float64,
                                                            name='weights')
                                        count=tf.placeholder(tf.float64,name='count')
                                        rui_tr=tf.transpose(rui,[1,2,0])
                                        rui_rs=tf.reshape(rui_tr,[-1,len(paths)])
                                        rui_sum=tf.reduce_sum(rui_rs*weights,axis=[1])
                                        
                                        rui_final=tf.reshape(rui_sum,[user_count,-1])

                                        diff=(rui_final-x_train)*x_mask
                                        square=tf.multiply(diff,diff)
                                        loss=tf.sqrt(tf.reduce_sum(square,axis=[0,1])/count)+\
                                              tf.pow(tf.norm(weights,ord=2),2)*(0.1)
                                        rmse=tf.sqrt(tf.reduce_sum(square,axis=[0,1])/count)

                                        train_step=tf.train.AdamOptimizer(learning_rate=parameters.learn_rate).\
                                                                        minimize(loss)
                                        
                                        init = tf.global_variables_initializer()
                                        sess.run(init)
                                        
                                        best_loss=1e99
                                        best_epoch=-1
                                        evaluate_every=5
                                        
                                        for epoch in range(1000):
                                                if epoch-best_epoch>500:
                                                        break

                                                feed_dict={x_train:score_train,
                                                           x_mask:mask_train,
                                                           count:non_0_count(mask_train)}
                                                sess.run(train_step,feed_dict)
                                                if epoch%evaluate_every==0 or epoch<evaluate_every:
                                                        feed_dict={x_train:score_train,
                                                                   x_mask:mask_train,
                                                                   count:non_0_count(mask_train)}
                                                        print(sess.run(weights,feed_dict=feed_dict))
    
                                                        dev_loss=sess.run(rmse,feed_dict=
                                                                          {x_train:score_dev,
                                                                           x_mask:mask_dev,
                                                                           count:non_0_count(mask_dev)})
                                                        if dev_loss<best_loss:
                                                                best_loss=dev_loss
                                                                best_epoch=epoch
##                                                                best_result=sess.run(result,feed_dict=
##                                                                          {user_train:user_dev,
##                                                                           item_train:item_dev,
##                                                                           y_train:score_dev})
                ##                                                save_path=tf.train.Saver().save(sess,
                ##                                                                     './models'+current_time+'/'+str(epoch)+'.ckpt')
                                                        print('epoch=',epoch,'\t',
                                                              'train=%.4f'%sess.run(loss,feed_dict),'\t',
                                                              'dev=%.4f'%dev_loss,'\t',
                                                              'best=%.4f'%best_loss,' @ ',best_epoch)
