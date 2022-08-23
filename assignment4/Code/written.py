import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def convert():
    path = "/mnt/c/Users/srava/Desktop/sem6/PRML - CS5691/assignment4/"
    folders = [path+"ai",path+"bA",path+"chA",path+"dA",path+"tA"]
    min = 1e9 + 0.0
    data_dict_train = {}
    data_dict_dev = {}

    for id,i in enumerate(folders):
        data_dict_train[id]=[]
        data_dict_dev[id]=[]
        for j in os.listdir(i+"/train/"):
            (filename,extension) = os.path.splitext(j)
            if extension == '.txt':
                temp_file=open(i+"/train/"+j,"r")
                temp = []
                # flag = True
                for id2,k in enumerate(temp_file.readlines()):
                    # print(k)
                    # temp = k[:-2].split(" ")
                    # print(temp)
                    l = list(map(float,k[:-2].split(" ")))
                    # if(flag):
                    #     print(l)
                    #     flag=False
                    if(min>l[0]):
                        min=l[0]
                    for k in range(int(l[0])):
                        temp.append([l[k*2+1],l[k*2+2]])
                    
                data_dict_train[id].append(temp)

        for j in os.listdir(i+"/dev/"):
            (filename,extension) = os.path.splitext(j)
            if extension == '.txt':
                temp_file=open(i+"/dev/"+j,"r")
                temp = []
                for id2,k in enumerate(temp_file.readlines()):
                    l = list(map(float,k[:-2].split(" ")))
                    if(min>l[0]):
                        min=l[0]
                    for k in range(int(l[0])):
                        temp.append([l[k*2+1],l[k*2+2]])
                    
                data_dict_dev[id].append(temp)
    
    for i in data_dict_train.keys():
        temp = []
        for k in range(len(data_dict_train[i])):
            scaler = MinMaxScaler()
            scaler.fit(data_dict_train[i][k])
            temp.append(scaler.transform(data_dict_train[i][k]))
        data_dict_train[i]=temp
    
    for i in data_dict_dev.keys():
        temp = []
        for k in range(len(data_dict_dev[i])):
            scaler = MinMaxScaler()
            scaler.fit(data_dict_dev[i][k])
            temp.append(scaler.transform(data_dict_dev[i][k]))
        data_dict_dev[i]=temp

    min = int(min)

    for i in data_dict_train.keys():
        temp1 = []
        for j in range(len(data_dict_train[i])):
            temp2 = []
            club = len(data_dict_train[i][j])-min+1
            for k in range(min):
                avg = []
                for l in range(club):
                    if(avg==[]):
                        avg=data_dict_train[i][j][k+l]
                    else:
                        for m in range(len(avg)):
                            avg[m]+=data_dict_train[i][j][k+l][m]

                for m in range(len(avg)):
                    avg[m]=avg[m]/club
                temp2.extend(avg)
            temp1.append(temp2)

        data_dict_train[i] = temp1

    for i in data_dict_dev.keys():
        temp1 = []
        for j in range(len(data_dict_dev[i])):
            temp2 = []
            club = len(data_dict_dev[i][j])-min+1
            for k in range(min):
                avg = []
                for l in range(club):
                    if(avg==[]):
                        avg=data_dict_dev[i][j][k+l]
                    else:
                        for m in range(len(avg)):
                            avg[m]+=data_dict_dev[i][j][k+l][m]

                for m in range(len(avg)):
                    avg[m]=avg[m]/club
                temp2.extend(avg)
            temp1.append(temp2)

        data_dict_dev[i] = temp1
    
    lst_train = []
    lst_dev = []
    map_train = []
    map_dev = []
    for i in data_dict_train.keys():
        map_train.append(i)
        lst_train.append(data_dict_train[i])
    for i in data_dict_dev.keys():
        map_dev.append(i)
        lst_dev.append(data_dict_dev[i])

    return data_dict_train,data_dict_dev,lst_train,lst_dev,map_train,map_dev
