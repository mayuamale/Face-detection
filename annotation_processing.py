import pandas as pd
import numpy as np

f = open('/content/gdrive/My Drive/face_trainval/wider_face_train_bbx_gt.txt','r')
frames = list()
column_names = ['x1','y1','w','h','blur','expression','illumination','invalid','occlusion','pose']
for i in range(0,12880):
  name = f.readline()
  #print(name)
  num = int(f.readline())
 # print(num)
  
  for j in range(0,num):
    temp = f.readline()
    temp = temp.split()
    temp = np.array(temp)
    
    if (j==0):
      temp1 = temp
      temp2 = [temp1]
      df = pd.DataFrame(temp2,columns=column_names)
    else:
      temp1 = np.vstack((temp1,temp))
      df = pd.DataFrame(temp1,columns=column_names)
    df['name'] = name
  frames.append(df)
    #print(df)
result = pd.concat(frames,ignore_index=True)
result['name'] = result['name'].apply(lambda x: x.rstrip('\n'))
result['name'] = result['name'].apply(lambda x: x.rsplit('/',1)[1])
result_final = result[['name','x1','y1','w','h']]
result_final['xmin'] = result_final['x1'].astype('float64')
result_final['xmax'] = result_final['x1'].astype('float64') + result_final['w'].astype('float64')
result_final['ymin'] = result_final['y1'].astype('float64')
result_final['ymax'] = result_final['y1'].astype('float64') + result_final['h'].astype('float64')
train = result_final.drop(columns=['x1','y1','w','h'])
cols = ['xmin', 'xmax','ymin','ymax']
train[cols] = train[cols].applymap(np.int64)
train.columns = ['ImageID','XMin','YMin','XMax','YMax']
train['Class'] = 'Man'

data = pd.DataFrame()                       #making annotation txt file
data['format'] = train['ImageID']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'content/gdrive/My Drive/All_images12' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['XMin'][i]) + ',' + str(train['YMin'][i]) + ',' + str(train['XMax'][i]) + ',' + str(train['YMax'][i]) + ',' + train['Class'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')