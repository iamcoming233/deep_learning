# Copyright 2016 The TensorFlow Authors. All Rights Reserved. 
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
# 
#     http://www.apache.org/licenses/LICENSE-2.0 
# 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# ============================================================================== 
"""Simple image classification with Inception. 
  
Run image classification with your model. 
  
This script is usually used with retrain.py found in this same 
directory. 
  
 This program creates a graph from a saved GraphDef protocol buffer, 
 and runs inference on an input JPEG image. You are required 
 to pass in the graph file and the txt file. 
  
 It outputs human readable strings of the top 5 predictions along with 
 their probabilities. 
  
 Change the --image_file argument to any jpg image to compute a 
 classification of that image. 
  
 Example usage: 
 python label_image.py --graph=retrained_graph.pb 
   --labels=retrained_labels.txt 
   --image=flower_photos/daisy/54377391_15648e8d18.jpg 
  
 NOTE: To learn to use this file and retrain.py, please see:
       ps:目前最大的问题是不能以batch输入进行预测。

 https://codelabs.developers.google.com/codelabs/tensorflow-for-poets 
 """ 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
 
import argparse
import sys
import os
import numpy as np
 
 
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument(
     '--imagedir', required=True ,type=str, help='Absolute path to image store dir.')

parser.add_argument(
     '--num_top_predictions', 
     type=int, 
     default=5, 
     help='Display this many predictions.') 
parser.add_argument(
     '--graph', 
     required=True, 
     type=str, 
     help='Absolute path to graph file (.pb)') 
parser.add_argument(
    '--labels', 
     required=True, 
     type=str, 
     help='Absolute path to labels file (.txt)') 
parser.add_argument(
     '--output_layer', 
     type=str, 
     default='final_result:0', 
     help='Name of the result operation')

parser.add_argument(
     '--input_layer', 
     type=str, 
     default='DecodeJpeg/contents:0',
     help='Name of the input operation')

parser.add_argument(
     '--summary',
     type=int,
     default=3,
     help='prediction summary option')

parser.add_argument(
     '--image_label',
     type=str,
     default='./label.txt',
     help='image map index(.txt)')

parser.add_argument(
     '--model',
     type=str,
     default='inception',
     help='model name')
 
 
def load_image_into_list(Imagedir):
   """Read in the image_data to be classified."""
   """upgrade by
      id:lidaiyuan
      修改内容:将原来的读取一张图片，改为将图像读入一个list 
   """
   Imagelist=[]
   for dirname,_,filenames in os.walk(Imagedir):
       for filename in filenames:
           filepath = dirname+'/'+filename
           Imagelist.append(filepath)
   return Imagelist

 
def load_labels(filename):
   """Read in labels, one label per line.""" 
   return [line.rstrip() for line in tf.gfile.GFile(filename)] 
 
 
 
 
def load_graph(filename):
   """Unpersists graph from file as default graph."""

   with tf.gfile.FastGFile(filename, 'rb') as f: 
     graph_def = tf.GraphDef() 
     graph_def.ParseFromString(f.read()) 
     tf.import_graph_def(graph_def, name='') 
 
def image_process(image_data,mean = 128,std = 128):
    #liadiyuan add:以下内容均获取自retrain.py 的预处理代码可以参考那边代码
    with tf.Session() as sess:
        if FLAGS.model=='mobilename':
         jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
         decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
         decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
         decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
         '''
         图像解码，并将原图像从rank 2 扩展为 rank 4
         '''
         resize_shape = tf.stack([299, 299])
         resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
         preprocessed = tf.image.resize_bilinear(decoded_image_4d,
                                                 resize_shape_as_int)
         ''''
        归一化图像大小
        '''
         offset_image = tf.subtract(preprocessed, mean)
         preprocessed = tf.multiply(offset_image, 1.0 / std)
         '''图像样本零均值，方差归一化？'''
         return sess.run(preprocessed,{jpeg_data:image_data})
        else:
         return image_data
def countCataNumber(tlist,labelindex):
    #该函数用于统计各类数量
    #param tlist imagelabel list
    #param labelindex:{labelname:labelindex}
    catalist=np.zeros(([len(labelindex.keys())]),dtype=np.int)
    for line in tlist:
        catalist[labelindex[line.split()[1]]]+=1
    return catalist
def ordinary_summary(labellist,imagelabel,predicationdict,flag=1,top_k=5):
    #lidaiyuan add:此函数的功能为：对预测结果进行统计并打印
    #包括每个类别的误报率，漏报率，每个类别被错误分类的最多的类别
    #param: labellist 标签列表文件路径
    #param：predicationlist 预测结果列表
    #param: top-k 打印top-k的置信度
    #param：标记设定打印和存储哪些内容，目前包括 1.- 打印每张图片的top5 的置信度.  2 - 测试样本每个类别被最多的错分类别统计(目前如果没有标签文件Img_label.txt则会报错)
    labelindex={}          #根据值
    count=0
    for l in labellist:
        labelindex[l]=count
        count+=1
    #创建标签与标签号的对应
    if flag>0:
        if os.path.exists('./prediction.txt'):
            os.remove('./prediction.txt')
        f=open('./prediction.txt','a')
        for key,predication in predicationdict.items():
            top_k = predication.argsort()[-5:][::-1]
            print ('%s 的top-5置信度为' % key)
            f.write('%s 的top-5置信度为:\n' % key)
            for node_id in top_k:
                human_string = labellist[node_id]
                score = predication[node_id]
                print('%s (score = %.5f)' % (human_string, score))
                f.write('%s (score = %.5f) \n' % (human_string, score))
            f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        f.close()
    if flag>1:
      if imagelabel==None:
          print('no label file!')
          return
      f = open(imagelabel)
      content=np.zeros((len(labellist),len(labellist)),dtype=np.int)  #预测和标签对应的矩阵 row:label col:predication
      text = f.readlines()
      catacount=countCataNumber(text,labelindex)
      for line in text:
        imagename = line.split()[0]                   #获取图片名
        label = line.split()[1]                       #获取对应图片的标签
        predication = predicationdict[imagename]      #获取图像的top-k预测
        top_k = predication.argsort()[-5:][::-1]
        Originlabel = labelindex[label]                #图像groundtrth
        predicationlabel = list(top_k)[0]  #没有设定阈值的情况下默认设定最大置信度对应的类别为预测类
        content[Originlabel][predicationlabel]+=1      #预测矩阵内容 +1
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-·统计结果-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~·')
      for key,value in labelindex.items():
        print('样本中 %s 类总量为 %d，其中错误预测的情况:' % (key,catacount[value]))
        for key_2, value_2 in labelindex.items():
            if key_2 == key:
                continue
            print('被误报为%s类的数量: %d'%(key_2,content[labelindex[key]][labelindex[key_2]]))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
def isconstraint(labellist,predication,constraints):
    #此函数用于判断预测值是否满足特定的约束
    #constraint：lable1:[{label1:con1,label2,con2},[label3:con3]]
    flag=False
    for constraint in constraints:
        flagand=True
        for l,c in constraint.items():
            top_k = predication.argsort()[-2:]
            try:
               labelindex = np.argwhere(top_k==labellist.index(l))[0]
               flagand = (flagand and predication[top_k[labelindex]]>c)
               if flagand == False:
                 break
            except Exception as e:
                print(e)
                flagand = False
                break
        flag = (flag or flagand)
    return flag
def spcialize_summary(labellist,predicationdict,reclassify,newimagelabeldir,newlabellistdir):
    #lidaiyuan add: 该函数主要用于对预测结果进行再分类，可以通过对通过对原始各类各种约束条件进行组合，形成新的类别，并进行统计
    #比如｛异常人脸：{口罩,墨镜}｝的组合,重新分类预测时时会按置信度降序排列(还没写完)
    #param:labellist  原始标签 to index dict
    #predicationdict  图像文件名 to 图像预测 dict
    #reclassify 进行再分类的置信度约束 dict{newlabel:[prelabel1:constraint...,prelabel:constraint]} 没有设定约束归为其他类，逻辑或条件使用list连接，需要同时满足的约束分开排列
    #                                 比如[prelabel:constraint,...,prelabel:constraint],prelabel:constraint
    #newimagelabeldir 图像 to newlabel文件路径
    #newlabellist     图像更新后的标签
    f=open(newimagelabeldir)
    newImaeglabel={}
    newlabellist = {}
    newpredication={}
    for line in f.readlines():
        newImaeglabel[line.split()[0]]=line.split()[1]
    f=open(newlabellistdir)
    for line in f.readlines():
        newlabellist[line.split()[1]]=line.split()[0]
    newlabelsum=len(newlabellist)
    predicterror = np.zeros([newlabelsum,newlabelsum])
    for key, predication in predicationdict.items():
        for newlabel,constraint in reclassify.items():
            groundtruth=np.zeros([newlabelsum])
            bytes=0
            flag=isconstraint(labellist,predication,constraint)
            labelindex=newlabellist[newlabel]
            if flag==True:
                groundtruth[int(labelindex)]=1
                newpredication[key]=groundtruth
        if flag==False:
            groundtruth[0] = 1  #groundtruth为one-hot编码标签，它的长度为类别的数目+1，for example:[0,1,0]代表第一类物体
            newpredication[key]=groundtruth
    for filename,label in newImaeglabel.items():
        predict=newpredication[filename].astype(np.int)
        tmp=np.zeros([newlabelsum],dtype=np.int)
        tmp[int(newlabellist[label])]=1
        if (tmp == predict).all():
            index = np.argwhere(predict == 1).squeeze(axis=0)[0]
            predicterror[index][index]+=1
        else:
            labelindex=newlabellist[label]
            index= np.argwhere(predict == 1).squeeze(axis=0)[0]
            predicterror[int(labelindex)][index]+=1
      #打印结果
    for i in range(newlabelsum):
         print('%s类'% list(newlabellist.keys())[i])
         for j in range(newlabelsum):
             print('误报为%s类的数量%d:'%(list(newlabellist.keys())[j],predicterror[i][j]))



def run_graph(image_dir_list, labels, input_layer_name, output_layer_name,
               num_top_predictions):
   with tf.Session() as sess: 
     # Feed the image_data as input to the graph. 
     #   predictions will contain a two-dimensional array, where one 
     #   dimension represents the input image count, and the other has 
     #   predictions per class

     #preprocess

     #preprocessed=sess.graph.get_tensor_by_name('input:0')

     #input_image=sess.run(resized_image,feed_dict={jpeg_data:image_data})
     #offset_image = tf.subtract(resized_image, input_mean)
     #mul_image = tf.multiply(offset_image, 1.0 / input_std)
     clresult = {}
     for image_dir in image_dir_list:
      decoded_image=image_process(tf.gfile.FastGFile(image_dir, 'rb').read())
      softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
      predictions, = sess.run(softmax_tensor, {input_layer_name: decoded_image})
      #imagefilename=image_dir.split('//')[1]
      imagefilename = os.path.basename(image_dir)#modify at 9-4
      #os.path.basename(image_dir)

      #lidaiyuan add:增加list储存预测结果
      clresult[imagefilename]=predictions
         # Sort to show labels in order of confidence
     #ldy add:create labellist to store the classification result:
     '''
     top_k = predictions.argsort()[-num_top_predictions:][::-1]
     for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
    '''

     return clresult
 
 
 
 
def main(argv):
   """Runs inference on an image.""" 
   if argv[1:]: 
     raise ValueError('Unused Command Line Args: %s' % argv[1:]) 
 
 
   if not tf.gfile.Exists(FLAGS.imagedir):
     tf.logging.fatal('dir does not exist %s', FLAGS.image)
 
 
   if not tf.gfile.Exists(FLAGS.labels): 
     tf.logging.fatal('labels file does not exist %s', FLAGS.labels) 

 
   if not tf.gfile.Exists(FLAGS.graph): 
     tf.logging.fatal('graph file does not exist %s', FLAGS.graph) 
 
 
   # load image 
   image_dir_list = load_image_into_list(FLAGS.imagedir)
 
 
   # load labels 
   labels = load_labels(FLAGS.labels) 
 
 
   # load graph, which is stored in the default session 
   load_graph(FLAGS.graph)
 
 
   clsdict=run_graph(image_dir_list, labels, FLAGS.input_layer, FLAGS.output_layer,
             FLAGS.num_top_predictions)

   # 2017-8-31:lidaiyuan add: make predication summary
   # 2017-9-1：prediction_summary modify to ordinary_summary
   ordinary_summary(labellist=labels,imagelabel=FLAGS.image_label,top_k=FLAGS.num_top_predictions,predicationdict=clsdict,flag=FLAGS.summary)

   spcialize_summary(labels, clsdict, {'unnormal':[{'mask':0.000,'sunglasses':0.000},{'mask':0.000,'hat':0.000},{'hat':0.000,'sunglasses':0.000},{'helmet':0.7}]}, 'D:/retrian/newlabel.txt', 'D:/retrian/newlabel_map_id.txt')

 
 
if __name__ == '__main__':
   FLAGS, unparsed = parser.parse_known_args() 
   tf.app.run(main=main, argv=sys.argv[:1]+unparsed) 
