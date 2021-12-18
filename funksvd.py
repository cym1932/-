import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
import embedding
import parameters

def sq_error(x_y, x, y, k, lamda):
        se=0
        indptr=x_y.indptr
        indices=x_y.indices
        data=x_y.data

        row=x_y.shape[0]
        col=x_y.shape[1] 
        x_index=0
        for i in range(indices.shape[0]):
                y_index=indices[i]
                while i>=indptr[x_index+1]:
                        x_index+=1
                value=data[i]
                product=0
                for d in range(k):
                        product=product+x[x_index,d]*y[y_index,d]
                        
                se=se+(value-product)*(value-product)
                
        for i in range(row):
                for j in range(k):
                        se=se+lamda*x[i,j]*x[i,j]
        for i in range(col):
                for j in range(k):
                        se=se+lamda*y[i,j]*y[i,j]
        return se        
                
def funksvd_sp(x_y, k=6, alpha=0.0008, lamda=0.1,
               decay=0.99, evaluate_every=10, moment=0):
        
        indptr=x_y.indptr
        indices=x_y.indices
        data=x_y.data

        row=x_y.shape[0]
        col=x_y.shape[1]
        
        x=np.random.rand(row,k)*2-1
        y=np.random.rand(col,k)*2-1

        square_error_old=sq_error(x_y,x,y,k,lamda)
        square_error=square_error_old
        print(square_error)
        x2=np.zeros((row,k))
        y2=np.zeros((col,k))

        n_iter=0
        while True: 
                x2=x2*moment
                y2=y2*moment
                x_index=0
                
                for i in range(indices.shape[0]):
                        y_index=indices[i]
                        while i>=indptr[x_index+1]:
                                x_index+=1
                        value=data[i]
                        
                        product=0
                        for d in range(k):
                                product=product+x[x_index,d]*y[y_index,d]
                       
                        am=alpha*(value-product)
                        for d in range(k):
                                x2[x_index,d]=x2[x_index,d]+am*y[y_index,d]
                                y2[y_index,d]=y2[y_index,d]+am*x[x_index,d]

                for i in range(row):
                        for j in range(k):
                                x[i,j]=x[i,j]*(1-alpha*lamda)+x2[i,j]
                for i in range(col):
                        for j in range(k):
                                y[i,j]=y[i,j]*(1-alpha*lamda)+y2[i,j]
                                

                n_iter+=1
                if n_iter%evaluate_every==0 or n_iter<evaluate_every:
                        if alpha>0.000001:
                                alpha=alpha*decay
                        square_error_old=square_error
                        square_error=sq_error(x_y,x,y,k,lamda)
                        print(n_iter,'\t',square_error)
                        if abs(-square_error+square_error_old)<0.001:
                                break
        
        return x,y,square_error,n_iter                       

def funksvd_tf(x_y, omega, k=6, alpha=0.0008, lamda=0.1, evaluate_every=20):
        graph=tf.Graph()
        row=x_y.shape[0]
        col=x_y.shape[1]
        count=0
        best_loss=1e99
        bestx=0
        besty=0
        with graph.as_default():
                sess=tf.Session()
                with sess.as_default():
                        param_lambda=tf.constant(lamda,dtype=tf.float64)
                        target=tf.constant(x_y.toarray(),dtype=tf.float64)
                        om=tf.constant(omega.toarray(),dtype=tf.float64)
                        x=tf.Variable(tf.random_uniform((row,k),minval=-1,maxval=1,dtype=tf.float64))
                        y=tf.Variable(tf.random_uniform((col,k),minval=-1,maxval=1,dtype=tf.float64))
                        x_mul_y=tf.matmul(x,y,transpose_b=True)
                        error=tf.multiply(om,tf.subtract(x_mul_y,target))
                        loss=tf.add(tf.pow(tf.norm(error,ord=2),2),
                        tf.multiply(param_lambda,tf.add(tf.pow(tf.norm(x,ord=2),2),
                                                        tf.pow(tf.norm(y,ord=2),2))))
                        
                        train_step=tf.train.AdamOptimizer(alpha).minimize(loss)

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
                                if n%evaluate_every==0:
                                        loss_new=sess.run(loss)
                                        print(n,loss_new)
                                        if loss_new<best_loss:
                                                best_loss=loss_new
                                                bestx=sess.run(x)
                                                besty=sess.run(y)
                                        if loss_new-loss_old>-0.012:
                                                count+=1
                                                if count>3:
                                                        break
                                        else:
                                                count-=2
                                                if count<0:
                                                        count=0
                                        loss_old=loss_new

                        #print(sess.run(x_mul_y))
                        return bestx,besty,n,best_loss
                        


##def funksvd_sp_sgd(x_y, n, m, k, alpha, lamda):
##        x=np.random.randn(n,k)
##        y=np.random.randn(n,k)
##
##        square_error_old=sq_error(x_y,x,y,n,m,k,lamda)
##        print(square_error_old)
##        square_error=square_error_old
##        num_iters=0
##        while True:
##                num_iters+=1
##                x2=np.zeros(k)
##                y2=np.zeros(k)
##                square_error_old=square_error
##                square_error=0
##
##                i=int(np.random.uniform(low=0.0, high=x_y.shape[0], size=None))
##
##                x_index=x_y[i,0]
##                y_index=x_y[i,1]
##                value=x_y[i,2]
##                
##                product=0
##                for d in range(k):
##                        product=product+x[x_index,d]*y[y_index,d]
##               
##                am=alpha*(value-product)
##
##                for d in range(k):
##                        x2[d]=am*y[y_index,d]
##                        y2[d]=am*x[x_index,d]
##                        
##                for d in range(k):
##                        x[x_index,d]=x[x_index,d]*(1-alpha*lamda)+x2[d]
##                        y[x_index,d]=y[x_index,d]*(1-alpha*lamda)+y2[d]
##
##                if alpha>0.00001:
##                        alpha=alpha*0.99
##                        
##                if num_iters%len(x_y)==0:         
##                        square_error_old=square_error
##                        square_error=sq_error(x_y,x,y,n,m,k,lamda)
##                        if abs(-square_error+square_error_old)<0.001:
##                                break
##                        print(square_error)
##
##        return x,y,square_error                         


if __name__ == '__main__':
        row=[]
        col=[]
        data=[]
        data2=[]
        dataset=parameters.dataset
        if dataset=='yelp':
                filename=parameters.folder_name+'user_business_short.txt'
        else:
                filename=parameters.folder_name+'a_user_item_short.txt'
                
        with open(filename,'r') as file:
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
                        data.append(a[2])
                        data2.append(1)
                        
        x_y=csr_matrix((data,(row,col)),shape=(maxi,maxj))
        indic=csr_matrix((data2,(row,col)),shape=(maxi,maxj))

        x,y,c,n=funksvd_tf(x_y,indic,alpha=parameters.learn_rate_embedding,lamda=parameters.l2_norm_pen)

##        x=x.toarray()
##        y=y.toarray()
##        


        if dataset=='yelp':
                filename=parameters.folder_name+'user_business_short_test.txt'
        else:
                filename=parameters.folder_name+'a_user_item_short_test.txt'
        count=0
        sq_error=0.0
        with open(filename,'r') as file:
                line=file.readline()
                while True:
                        line=file.readline()
                        if not line:
                                break
                        count+=1
                        a=[int(float(i)) for i in line.split()]
                        predict=np.sum(x[a[0]]*y[a[1]])
                        sq_error+=pow(predict-a[2],2)
                        print(pow(predict-a[2],2))

        print(sq_error)
        print(count)
        print(pow(sq_error/count,0.5))
