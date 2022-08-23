import numpy as np
import os

def convert():
    path = "Digits/"
    folders = [path+"1",path+"3",path+"4",path+"7",path+"9"]
    min = 1e9 + 0.0
    data_dict_train = {}
    data_dict_dev = {}

    for id,i in enumerate(folders):
        data_dict_train[id]=[]
        data_dict_dev[id]=[]
        for j in os.listdir(i+"/train/"):
            (filename,extension) = os.path.splitext(j)
            if extension == '.mfcc':
                temp_file=open(i+"/train/"+j,"r")
                temp = []
                for id2,k in enumerate(temp_file.readlines()):
                    # print(k)
                    l = list(map(float,k[1:-1].split(" ")))
                    if(id2==0):
                        if(min>l[1]):
                            min = l[1]
                    else:
                        temp.append(l)
                    # temp=np.array(temp)
                data_dict_train[id].append(temp)
        for j in os.listdir(i+"/dev/"):
            (filename,extension) = os.path.splitext(j)
            if extension == '.mfcc':
                temp_file=open(i+"/dev/"+j,"r")
                temp = []
                for id2,k in enumerate(temp_file.readlines()):
                    l = list(map(float,k[1:-1].split(" ")))
                    if(id2==0):
                        if(min>l[1]):
                            min = l[1]
                    else:
                        temp.append(l)
                    # temp=np.array(temp)
                data_dict_dev[id].append(temp)

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
        for ex in data_dict_train[i]:
          lst_train.append(ex)
          map_train.append(i)
    for i in data_dict_dev.keys():
      for ex in data_dict_dev[i]:
          lst_dev.append(ex)
          map_dev.append(i)

    return data_dict_train,data_dict_dev,lst_train,lst_dev,map_train,map_dev
