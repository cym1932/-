with open("user_business_short_full.txt",'r',encoding='utf8') as file:
    line=file.readline()
    user_count=int(line.split()[0])
    statistics=[0,0,0,0,0,0]
    global_mean=0
    global_count=0
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
