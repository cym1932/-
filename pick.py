import numpy as np
import parameters as pm
if __name__ == '__main__':
        if pm.dataset=='yelp':
                pick_size=2500
                number_of_business=209393

                indexs=np.random.permutation(number_of_business)[:pick_size]

                dict_bus={}
                dict_city={}
                dict_user={}
                dict_state={}
                dict_review={}
                dict_category={}

                i=0
                for index in indexs:
                        dict_bus[index]=i
                        i+=1

                city_count=0
                with open('./yelp_sorted/business_city.txt','r',encoding='utf8') as fp:
                        with open('./business_city_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(pick_size)+' 0\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_bus:
                                                city=int(data[1])
                                                if not city in dict_city:
                                                        dict_city[city]=city_count
                                                        city_count+=1
                                                        
                                                fout.write(str(dict_bus[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(dict_city[city]))
                                                fout.write('\t1\n')
                print('city_count=',city_count)

                state_count=0
                with open('./yelp_sorted/business_state.txt','r',encoding='utf8') as fp:
                        with open('./business_state_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(pick_size)+' 0\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_bus:
                                                state=int(data[1])
                                                if not state in dict_state:
                                                        dict_state[state]=state_count
                                                        state_count+=1
                                                        
                                                fout.write(str(dict_bus[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(dict_state[state]))
                                                fout.write('\t1\n')
                print('state_count=',state_count)
                                                  
                category_count=0
                with open('./yelp_sorted/business_category.txt','r',encoding='utf8') as fp:
                        with open('./business_category_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(pick_size)+' 0\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_bus:
                                                category=int(data[1])
                                                if not category in dict_category:
                                                        dict_category[category]=category_count
                                                        category_count+=1
                                                        
                                                fout.write(str(dict_bus[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(dict_category[category]))
                                                fout.write('\t1\n')
                print('category_count=',category_count)

                
                with open('./yelp_sorted/business_star.txt','r',encoding='utf8') as fp:
                        with open('./business_star_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(pick_size)+' 9\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_bus:
                                                star=int(data[1])
                                                fout.write(str(dict_bus[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(star))
                                                fout.write('\t1\n')

                user_count=0
                bu_count=0
                user_count_dict={}
                with open('./yelp_sorted/user_business.txt','r',encoding='utf8') as fp:
                        while True:
                                line=fp.readline()
                                if not line:
                                        break
                                data=line.split()
                                if int(data[1]) in dict_bus:
                                        user=int(data[0])
                                        if not user in user_count_dict:
                                                user_count_dict[user]=[int(data[1])]
                                        else:
                                                if not int(data[1]) in user_count_dict[user]:
                                                        user_count_dict[user].append(int(data[1]))

                duplicate=[]                                
                with open('./yelp_sorted/user_business.txt','r',encoding='utf8') as fp:
                        with open('./user_business_short_full.txt','w',encoding='utf8') as fout:
                                fout.write('0 '+str(pick_size)+'\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[1]) in dict_bus:
                                                user=int(data[0])
                                                if user in user_count_dict and len(user_count_dict[user])>=\
                                                   2:
                                                        if not user in dict_user:
                                                                dict_user[user]=user_count
                                                                user_count+=1
                                                        flag=dict_user[user]*pick_size+dict_bus[int(data[1])]
                                                        if not flag in duplicate:
                                                                duplicate.append(flag)
                                                                fout.write(str(dict_user[user]))
                                                                fout.write('\t')
                                                                fout.write(str(dict_bus[int(data[1])]))
                                                                fout.write('\t')
                                                                fout.write(str(float(data[2]))+'\n')
                                                                bu_count+=1
                print('user_count=',user_count)
                print('business_user_count=',bu_count)

                indices=np.random.permutation(bu_count)
                train_indices=indices[:(bu_count*4)//5]
                test_indices=indices[(bu_count*4)//5:]
                
                with open('./user_business_short_full.txt','r',encoding='utf8') as fp:
                        with open('./user_business_short.txt','w',encoding='utf8') as fout1:
                                with open('./user_business_short_test.txt','w',encoding='utf8') as fout2:
                                        line=fp.readline()
                                        fout1.write(str(user_count)+' '+str(pick_size)+'\n')
                                        fout2.write(str(user_count)+' '+str(pick_size)+'\n')
                                        index=0
                                        while True:
                                                line=fp.readline()
                                                if not line:
                                                        break                                           
                                                fout=0
                                                if index in train_indices:
                                                        fout=fout1
                                                else:
                                                        fout=fout2
                                                fout.write(line)
                                                index+=1


                uu_count=0
                with open('./yelp_sorted/user_user.txt','r',encoding='utf8') as fp:
                        with open('./user_user_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(user_count)+' '+str(user_count)+'\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_user and int(data[1]) in dict_user:
                                                fout.write(str(dict_user[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(dict_user[int(data[1])]))
                                                fout.write('\t1\n')
                                                uu_count+=1

                print('user_user_count=',uu_count)

                rev_count=0
                with open('./yelp_sorted/review_user.txt','r',encoding='utf8') as fp:
                        with open('./review_user_short.txt','w',encoding='utf8') as fout:
                                with open('./yelp_sorted/review_business.txt','r',encoding='utf8') as fp2:
                                        with open('./review_business_short.txt','w',encoding='utf8') as fout2:
                                                fout.write('0 '+str(pick_size)+'\n')
                                                fout2.write('0 '+str(user_count)+'\n')
                                                while True:
                                                        line=fp.readline()
                                                        line2=fp2.readline()
                                                        if not line:
                                                                break
                                                        if not line2:
                                                                break
                                                        data=line.split('\t')
                                                        data2=line2.split('\t')
                                                        review_id=data[0]
                                                        user_id=int(data[1])
                                                        business_id=int(data2[1])
                                                        if user_id in dict_user and business_id in dict_bus:
                                                                fout.write(str(rev_count))
                                                                fout.write('\t')
                                                                fout.write(str(dict_user[user_id]))
                                                                fout.write('\t1\n')
                                                                fout2.write(str(rev_count))
                                                                fout2.write('\t')
                                                                fout2.write(str(dict_bus[business_id]))
                                                                fout2.write('\t1\n')
                                                                rev_count+=1

                print('rev_count=',rev_count)
        
        else:
                pick_size=1000
                number_of_business=2753

                indexs=np.random.permutation(number_of_business)[:pick_size]

                dict_item={}
                dict_brand={}
                dict_user={}
                dict_state={}
                cdict_review={}
                dict_category={}

                i=0
                for index in indexs:
                        dict_item[index]=i
                        i+=1

                brand_count=0
                with open('./amazon_sorted/a_item_brand.txt','r',encoding='utf8') as fp:
                        with open('./a_item_brand_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(pick_size)+' 0\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_item:
                                                brand=int(data[1])
                                                if not brand in dict_brand:
                                                        dict_brand[brand]=brand_count
                                                        brand_count+=1
                                                        
                                                fout.write(str(dict_item[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(dict_brand[brand]))
                                                fout.write('\t1\n')
                print('brand_count=',brand_count)

##                category_count=0
##                with open('./yelp_sorted/business_state.txt','r',encoding='utf8') as fp:
##                        with open('./business_state_short.txt','w',encoding='utf8') as fout:
##                                fout.write(str(pick_size)+' 0\n')
##                                while True:
##                                        line=fp.readline()
##                                        if not line:
##                                                break
##                                        data=line.split('\t')
##                                        if int(data[0]) in dict_bus:
##                                                state=int(data[1])
##                                                if not state in dict_state:
##                                                        dict_state[state]=state_count
##                                                        state_count+=1
##                                                        
##                                                fout.write(str(dict_bus[int(data[0])]))
##                                                fout.write('\t')
##                                                fout.write(str(dict_state[state]))
##                                                fout.write('\t1\n')
##                print('state_count=',state_count)
##                                                  
                category_count=0
                with open('./amazon_sorted/a_item_category.txt','r',encoding='utf8') as fp:
                        with open('./a_item_category_short.txt','w',encoding='utf8') as fout:
                                fout.write(str(pick_size)+' 0\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[0]) in dict_item:
                                                category=int(data[1])
                                                if not category in dict_category:
                                                        dict_category[category]=category_count
                                                        category_count+=1
                                                        
                                                fout.write(str(dict_item[int(data[0])]))
                                                fout.write('\t')
                                                fout.write(str(dict_category[category]))
                                                fout.write('\t1\n')
                print('category_count=',category_count)

                
##                with open('./yelp_sorted/business_star.txt','r',encoding='utf8') as fp:
##                        with open('./business_star_short.txt','w',encoding='utf8') as fout:
##                                fout.write(str(pick_size)+' 9\n')
##                                while True:
##                                        line=fp.readline()
##                                        if not line:
##                                                break
##                                        data=line.split('\t')
##                                        if int(data[0]) in dict_bus:
##                                                star=int(data[1])
##                                                fout.write(str(dict_bus[int(data[0])]))
##                                                fout.write('\t')
##                                                fout.write(str(star))
##                                                fout.write('\t1\n')

                user_count=0
                bu_count=0
                user_count_dict={}
                with open('./amazon_sorted/a_user_item.txt','r',encoding='utf8') as fp:
                        while True:
                                line=fp.readline()
                                if not line:
                                        break
                                data=line.split()
                                if int(data[1]) in dict_item:
                                        user=int(data[0])
                                        if not user in user_count_dict:
                                                user_count_dict[user]=[int(data[1])]
                                        else:
                                                if not int(data[1]) in user_count_dict[user]:
                                                        user_count_dict[user].append(int(data[1]))

                duplicate=[]                                
                with open('./amazon_sorted/a_user_item.txt','r',encoding='utf8') as fp:
                        with open('./a_user_item_short_full.txt','w',encoding='utf8') as fout:
                                fout.write('0 '+str(pick_size)+'\n')
                                while True:
                                        line=fp.readline()
                                        if not line:
                                                break
                                        data=line.split('\t')
                                        if int(data[1]) in dict_item:
                                                user=int(data[0])
                                                if user in user_count_dict and len(user_count_dict[user])>=\
                                                   2 \
                                                   and user%5==0:
                                                        if not user in dict_user:
                                                                dict_user[user]=user_count
                                                                user_count+=1
                                                        flag=dict_user[user]*pick_size+dict_item[int(data[1])]
                                                        if not flag in duplicate:
                                                                duplicate.append(flag)
                                                                fout.write(str(dict_user[user]))
                                                                fout.write('\t')
                                                                fout.write(str(dict_item[int(data[1])]))
                                                                fout.write('\t')
                                                                fout.write(str(float(data[2]))+'\n')
                                                                bu_count+=1
                print('user_count=',user_count)
                print('item_user_count=',bu_count)

                indices=np.random.permutation(bu_count)
                train_indices=indices[:(bu_count*4)//5]
                test_indices=indices[(bu_count*4)//5:]
                
                with open('./a_user_item_short_full.txt','r',encoding='utf8') as fp:
                        with open('./a_user_item_short.txt','w',encoding='utf8') as fout1:
                                with open('./a_user_item_short_test.txt','w',encoding='utf8') as fout2:
                                        line=fp.readline()
                                        fout1.write(str(user_count)+' '+str(pick_size)+'\n')
                                        fout2.write(str(user_count)+' '+str(pick_size)+'\n')
                                        index=0
                                        while True:
                                                line=fp.readline()
                                                if not line:
                                                        break                                           
                                                fout=0
                                                if index in train_indices:
                                                        fout=fout1
                                                else:
                                                        fout=fout2
                                                fout.write(line)
                                                index+=1

##
##                uu_count=0
##                with open('./yelp_sorted/user_user.txt','r',encoding='utf8') as fp:
##                        with open('./user_user_short.txt','w',encoding='utf8') as fout:
##                                fout.write(str(user_count)+' '+str(user_count)+'\n')
##                                while True:
##                                        line=fp.readline()
##                                        if not line:
##                                                break
##                                        data=line.split('\t')
##                                        if int(data[0]) in dict_user and int(data[1]) in dict_user:
##                                                fout.write(str(dict_user[int(data[0])]))
##                                                fout.write('\t')
##                                                fout.write(str(dict_user[int(data[1])]))
##                                                fout.write('\t1\n')
##                                                uu_count+=1
##
##                print('user_user_count=',uu_count)
                                       
                                        
