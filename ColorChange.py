# -*- coding: UTF-8 -*-

import json
from PIL import Image,ImageDraw,ImageOps
from mpl_toolkits.mplot3d import Axes3D
import skimage
import numpy as np
import os
from skimage import io,color
from PIL import Image 
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.misc
import random
import csv
from lightness import *

def HSL_to_RGB(h,s,l):
    ''' Converts HSL colorspace (Hue/Saturation/Value) to RGB colorspace.
        Formula from http://www.easyrgb.com/math.php?MATH=M19#text19
       
        Input:
            h (float) : Hue (0...1, but can be above or below
                              (This is a rotation around the chromatic circle))
            s (float) : Saturation (0...1)    (0=toward grey, 1=pure color)
            l (float) : Lightness (0...1)     (0=black 0.5=pure color 1=white)
       
        Ouput:
            (r,g,b) (integers 0...255) : Corresponding RGB values
       
        Examples:
            >>> print HSL_to_RGB(0.7,0.7,0.6)
            (110, 82, 224)
            >>> r,g,b = HSL_to_RGB(0.7,0.7,0.6)
            >>> print g
            82
    '''
    def Hue_2_RGB( v1, v2, vH ):
        while vH<0.0: vH += 1.0
        while vH>1.0: vH -= 1.0
        if 6*vH < 1.0 : return v1 + (v2-v1)*6.0*vH
        if 2*vH < 1.0 : return v2
        if 3*vH < 2.0 : return v1 + (v2-v1)*((2.0/3.0)-vH)*6.0
        return v1
   
    if not (0 <= s <=1): raise ValueError,"s (saturation) parameter must be between 0 and 1."
    if not (0 <= l <=1): raise ValueError,"l (lightness) parameter must be between 0 and 1."
   
    r,b,g = (l*255,)*3
    if s!=0.0:
       if l<0.5 : var_2 = l * ( 1.0 + s )
       else     : var_2 = ( l + s ) - ( s * l )
       var_1 = 2.0 * l - var_2
       r = 255 * Hue_2_RGB( var_1, var_2, h + ( 1.0 / 3.0 ) )
       g = 255 * Hue_2_RGB( var_1, var_2, h )
       b = 255 * Hue_2_RGB( var_1, var_2, h - ( 1.0 / 3.0 ) )
      
    return (int(round(r)),int(round(g)),int(round(b)))


def RGB_to_HSL(r,g,b):

    if not (0 <= r <=255): raise ValueError,"r (red) parameter must be between 0 and 255."
    if not (0 <= g <=255): raise ValueError,"g (green) parameter must be between 0 and 255."
    if not (0 <= b <=255): raise ValueError,"b (blue) parameter must be between 0 and 255."
   
    var_R = r/255.0
    var_G = g/255.0
    var_B = b/255.0
   
    var_Min = min( var_R, var_G, var_B )    # Min. value of RGB
    var_Max = max( var_R, var_G, var_B )    # Max. value of RGB
    del_Max = var_Max - var_Min             # Delta RGB value
   
    l = ( var_Max + var_Min ) / 2.0
    h = 0.0
    s = 0.0
    if del_Max!=0.0:
       if l<0.5: s = del_Max / ( var_Max + var_Min )
       else:     s = del_Max / ( 2.0 - var_Max - var_Min )
       del_R = ( ( ( var_Max - var_R ) / 6.0 ) + ( del_Max / 2.0 ) ) / del_Max
       del_G = ( ( ( var_Max - var_G ) / 6.0 ) + ( del_Max / 2.0 ) ) / del_Max
       del_B = ( ( ( var_Max - var_B ) / 6.0 ) + ( del_Max / 2.0 ) ) / del_Max
       if    var_R == var_Max : h = del_B - del_G
       elif  var_G == var_Max : h = ( 1.0 / 3.0 ) + del_R - del_B
       elif  var_B == var_Max : h = ( 2.0 / 3.0 ) + del_G - del_R
       while h < 0.0: h += 1.0
       while h > 1.0: h -= 1.0
      
    return (h,s,l)

class backgroundColor():

    def getdatafitLab(self,data):

        wholeimage_backlight = []
        wholeimage_backcolor = []

        for eachdata in data:
            backL = eachdata['backL']
            backAb = eachdata['backAb']

            wholeimage_backlight.append(backL*1.0/100.0)
            wholeimage_backcolor.append([backAb[0]*1.0/128.0,backAb[1]*1.0/128.0])

        self.arraybackLight = np.array(wholeimage_backlight)[:, np.newaxis]
        params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(self.arraybackLight)
        #print grid.best_estimator_.bandwidth 
        self.wholeimage_backlight_kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(self.arraybackLight)

        self.arraybackcolor = np.array(wholeimage_backcolor)
        grid1 = GridSearchCV(KernelDensity(), params)
        grid1.fit(self.arraybackcolor)
        self.wholeimage_backcolor_kde = KernelDensity(kernel='gaussian', bandwidth=grid1.best_estimator_.bandwidth).fit(self.arraybackcolor)
        return self.wholeimage_backlight_kde,self.wholeimage_backcolor_kde

    def getdatafitHSV(self,data):

        wholeimage_backH = []
        wholeimage_backS = []
        wholeimage_backV = []

        for eachdata in data:
            backH = eachdata['backH']
            backS = eachdata['backS']
            backV = eachdata['backV']

            wholeimage_backH.append(backH)
            wholeimage_backS.append(backS)
            wholeimage_backV.append(backV)

        self.arraybackH = np.array(wholeimage_backH)[:, np.newaxis]
        params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(self.arraybackH)
        self.wholeimage_backH_kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(self.arraybackH)

        self.arraybackS = np.array(wholeimage_backS)[:, np.newaxis]
        grid1 = GridSearchCV(KernelDensity(), params)
        grid1.fit(self.arraybackS)
        self.wholeimage_backS_kde = KernelDensity(kernel='gaussian', bandwidth=grid1.best_estimator_.bandwidth).fit(self.arraybackS)

        self.arraybackV = np.array(wholeimage_backV)[:, np.newaxis]
        grid2 = GridSearchCV(KernelDensity(), params)
        grid2.fit(self.arraybackV)
        self.wholeimage_backV_kde = KernelDensity(kernel='gaussian', bandwidth=grid2.best_estimator_.bandwidth).fit(self.arraybackV)

        return self.wholeimage_backH_kde, self.wholeimage_backS_kde, self.wholeimage_backV_kde

    def getcontrastfitHSV(self,data):

        wholeimage_backContrastH = []
        wholeimage_backContrastS = []
        wholeimage_backContrastV = []

        for eachdata in data:
            backH = abs(eachdata['procH']-eachdata['backH']) 
            backS = abs(eachdata['procS']-eachdata['backS']) 
            backV = abs(eachdata['procV']-eachdata['backV'])

            wholeimage_backContrastH.append(backH)
            wholeimage_backContrastS.append(backS)
            wholeimage_backContrastV.append(backV)

        self.arraybackContrastH = np.array(wholeimage_backContrastH)[:, np.newaxis]
        #params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        #grid = GridSearchCV(KernelDensity(), params)
        #grid.fit(self.arraybackH)
        self.wholeimage_backContrastH_kde = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(self.arraybackContrastH)

        self.arraybackContrastS = np.array(wholeimage_backContrastS)[:, np.newaxis]
        self.wholeimage_backContrastS_kde = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(self.arraybackContrastS)

        self.arraybackContrastV = np.array(wholeimage_backContrastV)[:, np.newaxis]
        self.wholeimage_backContrastV_kde = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(self.arraybackContrastV)

        return self.wholeimage_backContrastH_kde, self.wholeimage_backContrastS_kde, self.wholeimage_backContrastV_kde

    def getBackgroundColorDistribution(self,data):

        wholeimage_backH = []
        wholeimage_backS = []
        wholeimage_backV = []

        for eachdata in data:
            backH = eachdata['backH']
            backS = eachdata['backS']
            backV = eachdata['backV']

            wholeimage_backH.append(backH)
            wholeimage_backS.append(backS)
            wholeimage_backV.append(backV)

        self.arraybackH = np.array(wholeimage_backH)[:, np.newaxis]
        self.arraybackS = np.array(wholeimage_backS)[:, np.newaxis]
        self.arraybackV = np.array(wholeimage_backV)[:, np.newaxis]

        random_state = 170

        hue_km = KMeans(n_clusters = 3 , random_state = random_state)
        hue_km.fit(self.arraybackH)
        self.hue_labels = hue_km.labels_
        self.hue_cluster_centers = hue_km.cluster_centers_
        total = len(self.hue_labels)
        sumNum = 0
        self.hue_percentage = []
        for i in range(len(self.hue_cluster_centers)):
            num = len(self.hue_labels[self.hue_labels<(i+1)])
            percenta = num - sumNum
            sumNum = num
            self.hue_percentage.append(percenta/float(total))

        sat_km = KMeans(n_clusters = 3 , random_state = random_state)
        sat_km.fit(self.arraybackS)
        self.sat_labels = sat_km.labels_
        self.sat_cluster_centers = sat_km.cluster_centers_
        total = len(self.sat_labels)
        sumNum = 0
        self.sat_percentage = []
        for i in range(len(self.sat_cluster_centers)):
            num = len(self.sat_labels[self.sat_labels<(i+1)])
            percenta = num - sumNum
            sumNum = num
            self.sat_percentage.append(percenta/float(total))

        val_km = KMeans(n_clusters = 3 , random_state = random_state)
        val_km.fit(self.arraybackV)
        self.val_labels = val_km.labels_
        self.val_cluster_centers = val_km.cluster_centers_
        total = len(self.val_labels)
        sumNum = 0
        self.val_percentage = []
        for i in range(len(self.val_cluster_centers)):
            num = len(self.val_labels[self.val_labels<(i+1)])
            percenta = num - sumNum
            sumNum = num
            self.val_percentage.append(percenta/float(total))

        return self.hue_cluster_centers, self.hue_percentage, self.sat_cluster_centers, self.sat_percentage, self.val_cluster_centers, self.val_percentage

    #def getdatafitHSVdistance(self,datapath):

class textColor():
    def __init__(self,data):

        backlight = []
        textlight = []
        textcolors = []
        backcolors = []

        backHue = []
        textHue = []
        textSat = []
        backSat = []
        textVal = []
        backVal = []
        for eachdata in data:
            text_backL = eachdata['text_backL']
            text_backAb = eachdata['text_backAb']
            textL = eachdata['textL']
            textAb = eachdata['textAb']

            backlight.append(text_backL*1.0/100)
            textlight.append(textL*1.0/100)
            textcolors.append([textAb[0]*1.0/128.0,textAb[1]*1.0/128.0])
            backcolors.append([text_backAb[0]*1.0/128.0,text_backAb[1]*1.0/128.0])

            textH = eachdata['textH']
            textS = eachdata['textS']
            textV = eachdata['textV']
            text_backH = eachdata['text_backH']
            text_backS = eachdata['text_backS']
            text_backV = eachdata['text_backV']

            backHue.append(text_backH)
            textHue.append(textH)
            textSat.append(textS)
            backSat.append(text_backS)
            textVal.append(textV)
            backVal.append(text_backV)

        self.arraybackLight = np.array(backlight)[:, np.newaxis]
        self.arraytextLight = np.array(textlight)[:, np.newaxis]

        self.arraybackcolor = np.array(backcolors)
        self.arraytextcolor = np.array(textcolors)

        self.arraybackHue = np.array(backHue)[:, np.newaxis]
        self.arraytextHue = np.array(textHue)[:, np.newaxis]

        self.arraytextSat = np.array(textSat)[:, np.newaxis]
        self.arraybackSat = np.array(backSat)[:, np.newaxis]

        self.arraytextVal = np.array(textVal)[:, np.newaxis]
        self.arraybackVal = np.array(backVal)[:, np.newaxis]

        random_state = 170

        light_km = KMeans(n_clusters = 10 , random_state = random_state)
        light_km.fit(self.arraytextLight)
        self.light_labels = light_km.labels_
        self.light_cluster_centers = light_km.cluster_centers_

        color_km = KMeans(n_clusters = 10 , random_state = random_state)
        color_km.fit(self.arraytextcolor)
        color_labels = color_km.labels_
        self.color_labels = np.array(color_labels)
        self.color_cluster_centers = color_km.cluster_centers_

        hue_km = KMeans(n_clusters = 10 , random_state = random_state)
        hue_km.fit(self.arraytextHue)
        self.hue_labels = hue_km.labels_
        self.hue_cluster_centers = hue_km.cluster_centers_

        sat_km = KMeans(n_clusters = 10 , random_state = random_state)
        sat_km.fit(self.arraytextSat)
        self.sat_labels = sat_km.labels_
        self.sat_cluster_centers = sat_km.cluster_centers_

        val_km = KMeans(n_clusters = 10 , random_state = random_state)
        val_km.fit(self.arraytextVal)
        self.val_labels = val_km.labels_
        self.val_cluster_centers = val_km.cluster_centers_


    def getdatafit(self,l,a,b):

        self.inputlight = [l]
        self.inputlight = np.array(self.inputlight).reshape((1,-1))
        self.inputcolor = [a,b]
        self.inputcolor = np.array(self.inputcolor).reshape((1,-1))

        classifiers = {'L1 logistic': LogisticRegression(C=1.0, penalty='l1')
            }

        for index, (name, classifier) in enumerate(classifiers.items()):
            
            classifier.fit(self.arraybackLight, self.light_labels)
            y_pred = classifier.predict(self.inputlight)
            probas = classifier.predict_proba(self.inputlight)
            
            preData = []
            for i in range(10):
                for j in range(int(probas[0,i]*1000)):
                    preData.append(self.light_cluster_centers[i,0])
            preData = np.array(preData)[:, np.newaxis]

            params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(preData)
            self.text_light_kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(preData)
            
            
            
            classifier.fit(self.arraybackcolor, self.color_labels)
            y_pred = classifier.predict(self.inputcolor)
            probas = classifier.predict_proba(self.inputcolor)
            #print probas
            
            preData = []
            for i in range(10):
                for j in range(int(probas[0,i]*1000)):
                    preData.append(self.color_cluster_centers[i])
            preData = np.array(preData)

            grid1 = GridSearchCV(KernelDensity(), params)
            grid1.fit(preData)
            self.text_color_kde = KernelDensity(kernel='gaussian', bandwidth=grid1.best_estimator_.bandwidth).fit(preData)

        return self.text_light_kde, self.text_color_kde,probas,self.light_cluster_centers

    def getdatafitHSV(self,h,s,v):

        self.inputH = [h]
        self.inputH = np.array(self.inputH).reshape((1,-1))
        self.inputS = [s]
        self.inputS = np.array(self.inputS).reshape((1,-1))
        self.inputV = [v]
        self.inputV = np.array(self.inputV).reshape((1,-1))

        classifiers = {'L1 logistic': LogisticRegression(C=1.0, penalty='l1')
            }

        for index, (name, classifier) in enumerate(classifiers.items()):
            
            classifier.fit(self.arraybackHue, self.hue_labels)
            y_pred = classifier.predict(self.inputH)
            probas = classifier.predict_proba(self.inputH)
            
            preData = []
            for i in range(10):
                for j in range(int(probas[0,i]*1000)):
                    preData.append(self.hue_cluster_centers[i,0])
            preData = np.array(preData)[:, np.newaxis]

            params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(preData)
            self.text_hue_kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(preData)
            
            
            
            classifier.fit(self.arraybackSat, self.sat_labels)
            y_pred = classifier.predict(self.inputS)
            probas = classifier.predict_proba(self.inputS)
            #print probas
            
            preData = []
            for i in range(10):
                for j in range(int(probas[0,i]*1000)):
                    preData.append(self.sat_cluster_centers[i])
            preData = np.array(preData)

            grid1 = GridSearchCV(KernelDensity(), params)
            grid1.fit(preData)
            self.text_sat_kde = KernelDensity(kernel='gaussian', bandwidth=grid1.best_estimator_.bandwidth).fit(preData)

            classifier.fit(self.arraybackVal, self.val_labels)
            y_pred = classifier.predict(self.inputV)
            probas = classifier.predict_proba(self.inputV)
            #print probas
            
            preData = []
            for i in range(10):
                for j in range(int(probas[0,i]*1000)):
                    preData.append(self.val_cluster_centers[i])
            preData = np.array(preData)

            grid2 = GridSearchCV(KernelDensity(), params)
            grid2.fit(preData)
            self.text_val_kde = KernelDensity(kernel='gaussian', bandwidth=grid2.best_estimator_.bandwidth).fit(preData)

        return self.text_hue_kde, self.text_sat_kde, self.text_val_kde

    def getTextColorDistribution(self):

        total = len(self.hue_labels)
        sumNum = 0
        self.hue_percentage = []
        for i in range(len(self.hue_cluster_centers)):
            num = len(self.hue_labels[self.hue_labels<(i+1)])
            percenta = num - sumNum
            sumNum = num
            self.hue_percentage.append(percenta/float(total))

        total = len(self.sat_labels)
        sumNum = 0
        self.sat_percentage = []
        for i in range(len(self.sat_cluster_centers)):
            num = len(self.sat_labels[self.sat_labels<(i+1)])
            percenta = num - sumNum
            sumNum = num
            self.sat_percentage.append(percenta/float(total))

        total = len(self.val_labels)
        sumNum = 0
        self.val_percentage = []
        for i in range(len(self.val_cluster_centers)):
            num = len(self.val_labels[self.val_labels<(i+1)])
            percenta = num - sumNum
            sumNum = num
            self.val_percentage.append(percenta/float(total))

        return self.hue_cluster_centers, self.hue_percentage, self.sat_cluster_centers, self.sat_percentage, self.val_cluster_centers, self.val_percentage


def getImageFromJson(filename):
    banner = json.load(open(filename , 'r'))
    children = banner['children']
    banner_height = banner['height']
    banner_width = banner['width']
    childrens = []
    for item in children:
        child = [item['top'] , item['left'] , item['width'] , item['height'] , item['label'] , item['src'] , item['zIndex']]
        childrens.append(child)

    return childrens,banner_width,banner_height

def paint(childrens,w,h):
    img = Image.new('RGB' , (w , h))
    childrens.sort(lambda a,b:a[6]-b[6])
    for item in childrens:
        #print item
        url = item[5]
        filename = url.split('/')
        filename = filename[-3]+'_'+ filename[-2]+'_'+filename[-1]
        
        elem = Image.open('./banner_elements/change_elements/'+ filename)
        elem = elem.resize((item[2] , item[3]) , Image.ANTIALIAS)
        
        if elem.mode == 'RGBA':
            img.paste(elem , (item[1] , item[0]) , mask = elem)
        else:
            img.paste(elem , (item[1] , item[0]))

    return img

def imgcolor_h_change(filename, target):

    #img = Image.open('./banner_elements/original_elements/'+filename)

    #亮度调整
    back_rgb_color = [[[target[0]/255.0, target[1]/255.0, target[2]/255.0]]] 
    back_lab_color = color.rgb2lab(back_rgb_color,illuminant='D65', observer='2')
    light = back_lab_color[0][0][0]*1.0/100.0 
    lightnessAdjust(light,'./banner_elements/original_elements/'+filename)


    colors=[]
    colors.append(target)
    img = Image.open("./result.png")
    #print(img.shape)

    pp=Mapping(img,colors,20)
    
    if len(colors)>1:
        img_recolor=pp.recolor()
        
        #save_theme('res/'+str(test_id)+'_gradation.png',pp.gradation_rgb,1000,100)
        img.save('res/'+str(test_id)+'_img.png')
        img_recolor.save('res/'+str(test_id)+'_res.png')
    else:
        #save_theme('res/'+str(test_id)+'_gradation.png',pp.palette,1000,100)
        #img.save('res/'+str(test_id)+'_img.png')

        #img_recolor=pp.recolor_SingleColor()
        #img_recolor.save('res/'+str(test_id)+'_res_l2p.png')

        #img_recolor2=pp.recolor_SingleColor2()  
        #img_recolor2.save('res/'+str(test_id)+'_res_p2p.png')

        img_recolor3=pp.recolor_SingleColor3()  
        img_recolor3.save('./banner_elements/change_elements/'+filename)


def colorDistance(sourceColor, targetColor):

    sourceColor = np.array(sourceColor)
    dists = []

    for each in targetColor:
        each = np.array(each)  
        dist = np.sqrt(np.sum(np.square(sourceColor - each))) 
        dists.append(dist)

    if len(dists) == 0:
        return 500
    else:
        return min(dists)


if __name__ == '__main__':

    template = './banner_template/template5.json'
    #template = './banner_template/template5-black.json' #19,15,18
    #template = './banner_template/template5-red.json' #108,1,2
    #template = './banner_template/template5-green.json' #1,175,139
    #template = './banner_template/template5-yellow.json' #240,203,92
    #template = './banner_template/template5-pink.json' #226,165,152
    #template = './banner_template/template5-blue.json' #1,52,161

    csvFile = open("./colorForBanner.csv", "r")
    reader = csv.reader(csvFile)
    backgroundData = []
    textData = []

    keywords = "女装"
    productcolor = [0,0,0]

    for item in reader:

        eachBackgroundData = {}
        filename = item[0]
        category = item[1].decode("gbk").encode("utf-8")
        description = item[2].decode("gbk").encode("utf-8")

        productColor = json.loads(item[5])

        #if colorDistance(productcolor,productColor)<40:
        if colorDistance(productcolor,productColor)<40 and category == keywords:
        #if category == keywords:
            #print filename
        #if keywords in description:
            backgroundcolors = json.loads(item[3])
            
            #-----------背景色采集------------
            BackR = backgroundcolors["imgBackgroundcolor_R"]
            BackG = backgroundcolors["imgBackgroundcolor_G"]
            BackB = backgroundcolors["imgBackgroundcolor_B"]         
            h,s,v = RGB_to_HSL(BackR,BackG,BackB)
            #CIELab channel
            rgb_color = [[[BackR/255.0, BackG/255.0, BackB/255.0]]]  # parameter between 0-1
            lab_color = color.rgb2lab(rgb_color,illuminant='D65', observer='2')
            l = lab_color[0][0][0] # 0-100
            a = lab_color[0][0][1] # -128 - 128
            b = lab_color[0][0][2] # -128 - 128
            
            eachBackgroundData['backH']= h
            eachBackgroundData['backS']= s
            eachBackgroundData['backV']= v
            eachBackgroundData['backL']= l
            eachBackgroundData['backAb']= [a,b]
            #eachdata['textColor']= textColor
            backgroundData.append(eachBackgroundData)
            
            #-------------字体色采集---------------
            textColors = json.loads(item[4])
            for eachText in textColors:
                eachTextData = {}
                textR = eachText["textcolor_R"]
                textG = eachText["textcolor_G"]
                textB = eachText["textcolor_B"]
                textbackground_R = eachText["backgroundcolor_R"]
                textbackground_G = eachText["backgroundcolor_G"]
                textbackground_B = eachText["backgroundcolor_B"]
                
                if textR < 0:
                    textR = 0
                if textG < 0:
                    textG = 0
                if textB < 0:
                    textB = 0
                if textR > 255:
                    textR = 255
                if textG > 255:
                    textG = 255
                if textB > 255:
                    textB = 255
                
                if textbackground_R < 0:
                    textbackground_R = 0
                if textbackground_G < 0:
                    textbackground_G = 0
                if textbackground_B < 0:
                    textbackground_B = 0
                if textbackground_R > 255:
                    textbackground_R = 255
                if textbackground_G > 255:
                    textbackground_G = 255
                if textbackground_B > 255:
                    textbackground_B = 255
                
                
                bh,bs,bv = RGB_to_HSL(textbackground_R,textbackground_G,textbackground_B)
                th,ts,tv = RGB_to_HSL(textR,textG,textB)
                #CIELab channel
                rgb_color = [[[textbackground_R/255.0, textbackground_G/255.0, textbackground_B/255.0]]]  # parameter between 0-1
                lab_color = color.rgb2lab(rgb_color,illuminant='D65', observer='2')
                bl = lab_color[0][0][0] # 0-100
                ba = lab_color[0][0][1] # -128 - 128
                bb = lab_color[0][0][2] # -128 - 128
                rgb_color = [[[textR/255.0, textG/255.0, textB/255.0]]]  # parameter between 0-1
                lab_color = color.rgb2lab(rgb_color,illuminant='D65', observer='2')
                tl = lab_color[0][0][0] # 0-100
                ta = lab_color[0][0][1] # -128 - 128
                tb = lab_color[0][0][2] # -128 - 128

                eachTextData['text_backH']= bh
                eachTextData['text_backS']= bs
                eachTextData['text_backV']= bv
                eachTextData['text_backL']= bl
                eachTextData['text_backAb']= [ba,bb]
                eachTextData['textH']= th
                eachTextData['textS']= ts
                eachTextData['textV']= tv
                eachTextData['textL']= tl
                eachTextData['textAb']= [ta,tb]
                textData.append(eachTextData)
                
    print "done" 

    backgroundColor = backgroundColor()
    textColor = textColor(textData)

    #wholeimage_backlight_kde, wholeimage_backcolor_kde = backgroundColor.getdatafitLab(backgroundData)
    wholeimage_backH_kde, wholeimage_backS_kde, wholeimage_backV_kde = backgroundColor.getdatafitHSV(backgroundData)

    hue_cluster_centers, hue_percentage, sat_cluster_centers, sat_percentage, val_cluster_centers, val_percentage = backgroundColor.getBackgroundColorDistribution(backgroundData)

    total_score = []
    total_color = []

    #print val_cluster_centers,val_percentage

    for k in range(100):

        randomProbability = random.random()
        sum = 0
        for l in range (len(hue_percentage)):
            sum = sum +  hue_percentage[l]
            if randomProbability < sum:
                break
        hue_random = np.random.normal(loc=hue_cluster_centers[l][0], scale=0.1, size=None)
        if hue_random > 1:
            hue_random = hue_random-1
        if hue_random < 0:
            hue_random = 1-hue_random

        randomProbability = random.random()
        sum = 0
        for l in range (len(sat_percentage)):
            sum = sum +  sat_percentage[l]
            if randomProbability < sum:
                break
        sat_random = np.random.normal(loc=sat_cluster_centers[l][0], scale=0.1, size=None)
        if sat_random > 1:
            sat_random = 1
        if sat_random < 0:
            sat_random = 0

        randomProbability = random.random()
        sum = 0
        for l in range (len(val_percentage)):
            sum = sum +  val_percentage[l]
            if randomProbability < sum:
                break
        val_random = np.random.normal(loc=val_cluster_centers[l][0], scale=0.1, size=None)
        if val_random > 1:
            val_random = 1
        if val_random < 0:
            val_random = 0

        #print sat_random
        r,g,b = HSL_to_RGB(hue_random,sat_random,val_random)
        #print r,g,b

        '''
        back_rgb_color = [[[r/255.0, g/255.0, b/255.0]]] 
        back_lab_color = color.rgb2lab(back_rgb_color,illuminant='D65', observer='2')
        bl = back_lab_color[0][0][0]*1.0/100.0
        ba = back_lab_color[0][0][1]*1.0/128.0
        bb = back_lab_color[0][0][2]*1.0/128.0

        back_light_score = np.exp(wholeimage_backlight_kde.score_samples(bl))
        back_color_score = np.exp(wholeimage_backcolor_kde.score_samples(np.array([ba,bb]).reshape((1,-1))))
        total_score.append(back_light_score+back_color_score)
        '''

        h,s,v = RGB_to_HSL(r,g,b)
        back_h_score = np.exp(wholeimage_backH_kde.score_samples(h))
        back_s_score = np.exp(wholeimage_backS_kde.score_samples(s))
        back_v_score = np.exp(wholeimage_backV_kde .score_samples(v))
        total_score.append(back_h_score+back_s_score+back_v_score)

        selected_color = {}
        selected_color['background'] = [r,g,b]
        #selected_color['text'] = palette[j]
        total_color.append(selected_color)

    score_copy = list(total_score)
    score_copy.sort(reverse=True)
    #print total_color

    #text优化
    hue_cluster_centers, hue_percentage, sat_cluster_centers, sat_percentage, val_cluster_centers, val_percentage = textColor.getTextColorDistribution()

    for i in range(0,4):
        index = total_score.index(score_copy[i])

        print i,total_color[index]['background']

        total_score_text = []
        total_color_text = []

        h,s,l = RGB_to_HSL(total_color[index]['background'][0],total_color[index]['background'][1],total_color[index]['background'][2])

        text_hue_kde, text_sat_kde, text_val_kde = textColor.getdatafitHSV(h,s,l)
        '''
        back_rgb_color = [[[total_color[index]['background'][0]/255.0, total_color[index]['background'][1]/255.0, total_color[index]['background'][2]/255.0]]] 
        back_lab_color = color.rgb2lab(back_rgb_color,illuminant='D65', observer='2')
        bl = back_lab_color[0][0][0]*1.0/100.0
        ba = back_lab_color[0][0][1]*1.0/128.0
        bb = back_lab_color[0][0][2]*1.0/128.0

        text_light_kde, text_color_kde,probas,light_cluster_centers = textColor.getdatafit(bl,ba,bb)
        '''

        for k in range(200):
            '''
            randomProbability = random.random()
            sum = 0
            for l in range (len(hue_percentage)):
                sum = sum +  hue_percentage[l]
                if randomProbability < sum:
                    break
            hue_random = np.random.normal(loc=hue_cluster_centers[l], scale=0.1, size=None)
            if hue_random > 1:
                hue_random = 1
            if hue_random < 0:
                hue_random = 0
            '''
            randomProbability = random.random()
            sum = 0
            for l in range (len(hue_percentage)):
                sum = sum +  hue_percentage[l]
                if randomProbability < sum:
                    break
            hue_random = np.random.normal(loc=hue_cluster_centers[l][0], scale=0.1, size=None)
            if hue_random > 1:
                hue_random = hue_random-1
            if hue_random < 0:
                hue_random = 1-hue_random

            
            randomProbability = random.random()
            sum = 0
            for l in range (len(sat_percentage)):
                sum = sum +  sat_percentage[l]
                if randomProbability < sum:
                    break
            sat_random = np.random.normal(loc=sat_cluster_centers[l][0], scale=0.1, size=None)
            if sat_random > 1:
                sat_random = 1
            if sat_random < 0:
                sat_random = 0
            

            randomProbability = random.random()
            sum = 0
            for l in range (len(val_percentage)):
                sum = sum +  val_percentage[l]
                if randomProbability < sum:
                    break
            val_random = np.random.normal(loc=val_cluster_centers[l][0], scale=0.1, size=None)
            if val_random > 1:
                val_random = 1
            if val_random < 0:
                val_random = 0

            #print sat_random
            r,g,b = HSL_to_RGB(hue_random,sat_random,val_random)
            #print r,g,b

            '''
            back_rgb_color = [[[r/255.0, g/255.0, b/255.0]]]
            text_lab_color = color.rgb2lab(back_rgb_color,illuminant='D65', observer='2')
            tl = text_lab_color[0][0][0]*1.0/100.0
            ta = text_lab_color[0][0][1]*1.0/128.0
            tb = text_lab_color[0][0][2]*1.0/128.0
            text_light_score = np.exp(text_light_kde.score_samples(tl))
            text_color_score = np.exp(text_color_kde.score_samples(np.array([ta,tb]).reshape((1,-1))))
            total_score_text.append(text_light_score+text_color_score)
            '''

            h,s,v = RGB_to_HSL(r,g,b)
            text_hue_score = np.exp(text_hue_kde.score_samples(h))
            text_sat_score = np.exp(text_sat_kde.score_samples(s))
            text_val_score = np.exp(text_val_kde.score_samples(v))       
            total_score_text.append(text_hue_score+text_sat_score+text_val_score)

            selected_color = {}
            selected_color['background'] = [total_color[index]['background'][0],total_color[index]['background'][1],total_color[index]['background'][2]]
            selected_color['text'] = [r,g,b]
            total_color_text.append(selected_color)

        score_copy_text = list(total_score_text)
        score_copy_text.sort(reverse=True)
        #print total_color_text

        for p in range(1):
            index11 = total_score_text.index(score_copy_text[p])

            print p,total_color_text[index11]
           
            childrens,banner_width,banner_height = getImageFromJson(template)
            for item in childrens:
                if item[4] == "background":
                    url = item[5]
                    background_filename = url.split('/')
                    background_filename = background_filename[-3]+'_'+ background_filename[-2]+'_'+background_filename[-1]
                    
                    imgcolor_h_change(background_filename,total_color_text[index]['background'])
                if item[4].startswith("title"):
                    url = item[5]
                    text_filename = url.split('/')
                    text_filename = text_filename[-3]+'_'+ text_filename[-2]+'_'+text_filename[-1]
                    print i
                    print total_color_text[index]['text']
                    
                    imgcolor_h_change(text_filename,total_color_text[index]['text']) 
            img = paint(childrens,banner_width,banner_height)
            img.save('./recoloring_randomSample_text/'+keywords+str(i)+'_'+str(p)+'.jpg')
            