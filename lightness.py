# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image 
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from skimage import color
from color2 import *
import math
import hsl

def RGB_to_HSL(r,g,b):
    ''' Converts RGB colorspace to HSL (Hue/Saturation/Value) colorspace.
        Formula from http://www.easyrgb.com/math.php?MATH=M18#text18
       
        Input:
            (r,g,b) (integers 0...255) : RGB values
       
        Ouput:
            (h,s,l) (floats 0...1): corresponding HSL values
       
        Example:
            >>> print RGB_to_HSL(110,82,224)
            (0.69953051643192476, 0.69607843137254899, 0.59999999999999998)
            >>> h,s,l = RGB_to_HSL(110,82,224)
            >>> print s
            0.696078431373
    '''
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


def lightnessAdjust(lightness,name):

    targetLight = lightness

    img = Image.open(name)
    #img = img.convert('RGB')

    im = np.array(img)

    filterColor = []
    for i in range(len(im[:,1])):
        for j in range(len(im[1,:])):
            if im[i,j][3] != 0 :
                h,s,v = RGB_to_HSL(im[i,j][0], im[i,j][1], im[i,j][2])
                filterColor.append(v)
                #filterColor.append([im[i,j][0]*1.0/255, im[i,j][1]*1.0/255, im[i,j][2]*1.0/255])

    Hinput = np.array(filterColor)[:, np.newaxis]
    #Hinput = np.array(filterColor)

    ms = MeanShift(bandwidth=0.1, bin_seeding=True)
    ms.fit(Hinput)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #print cluster_centers
    #print("number of estimated clusters : %d" % n_clusters_)
    total = len(labels)
    sumNum = 0
    percentage = []
    eachdata = {}
    for i in range(n_clusters_):
        num = len(labels[labels<(i+1)])
        percenta = num - sumNum
        sumNum = num
        percentage.append(percenta/float(total))
    #print percentage

    deltaV = targetLight-cluster_centers[0]
    deltaVR = targetLight-(1-cluster_centers[0])
    minV = min(Hinput)
    maxV = max(Hinput)
    boundry = max(abs(cluster_centers[0] - minV),abs(maxV - cluster_centers[0]))
    #print maxV,minV
    #print deltaV,boundry
    #print deltaVR

    if abs(deltaV) < abs(deltaVR):
    #if 1:
        for i in range(len(im[:,1])):
            for j in range(len(im[1,:])):
                if im[i,j][3] != 0 :
                    h,s,v = RGB_to_HSL(im[i,j][0], im[i,j][1], im[i,j][2])
                    rate = (boundry - abs(v-cluster_centers[0]))/boundry
                    v = v + deltaV * rate
                    #print v
                    if v > 1:
                        v = 1
                    elif v < 0:
                        v = 0
                    r,g,b = HSL_to_RGB(h,s,v)
                    im[i][j][0] = r
                    im[i][j][1] = g
                    im[i][j][2] = b
    else:
        print "reverse"
        for i in range(len(im[:,1])):
            for j in range(len(im[1,:])):
                if im[i,j][3] != 0 :
                    h,s,v = RGB_to_HSL(im[i,j][0], im[i,j][1], im[i,j][2])
                    v = 1- v
                    locate = 1-cluster_centers[0]
                    rate = (boundry - abs(v-locate))/boundry
                    #print rate
                    v = v + deltaVR * rate
                    if v > 1:
                        v = 1
                    elif v < 0:
                        v = 0
                    r,g,b = HSL_to_RGB(h,s,v)
                    im[i][j][0] = r
                    im[i][j][1] = g
                    im[i][j][2] = b

    scipy.misc.imsave('./result.png', im)

class Mapping(object):
    def __init__(self,img,palette,num):
        #print 'mode',img.mode
        self.img_rgba=img
        self.img_copy = self.img_rgba.copy()
        img_rgb=img.convert('RGB')
        self.img = img_rgb
        
        self.dataArray = list(self.img_rgba.getdata()) 
        self.data_lab=[]
        for i in range(len(self.dataArray)):
            self.data_lab.append(rgb2lab([self.dataArray[i][0],self.dataArray[i][1],self.dataArray[i][2]]))
        self.palette=palette
        self.gradation_num=num
        self.bin_range = 16;
        self.bin_size = 256 / self.bin_range;

    # line1 is the origin regression line
    # line2 is the template regression line
    def getRotateMat(self,line1,line2):
        angle1=np.arctan(line1[0])
        angle2=np.arctan(line2[0])
        angle=angle2-angle1
        mat=np.zeros((3,3))
        mat[0][0]=np.cos(angle)
        mat[0][1]=-np.sin(angle)
        mat[0][2]=0
        mat[1][0]=np.sin(angle)
        mat[1][1]=np.cos(angle)
        mat[1][2]=0
        mat[2][0]=mat[2][1]=0
        mat[2][2]=1
        return mat
    def getRotateMat_angle(self,angle1,angle2):
        angle=angle2-angle1
        mat=np.zeros((3,3))
        mat[0][0]=np.cos(angle)
        mat[0][1]=-np.sin(angle)
        mat[0][2]=0
        mat[1][0]=np.sin(angle)
        mat[1][1]=np.cos(angle)
        mat[1][2]=0
        mat[2][0]=mat[2][1]=0
        mat[2][2]=1
        return mat
    # y=alpha*x+beta
    # mark为True，则是将直线平移到原点
    # mark为False，则是将直线从原点平移到所在位置
    def getTranslateMat(self,line,mark):
        alpha=line[0]
        beta=line[1]
        if mark==True:
            if abs(alpha)<=1:
                mat=[[1,0,0],[0,1,-beta],[0,0,1]]
            else:
                mat=[[1,0,beta*1.0/alpha],[0,1,0],[0,0,1]]
        else:
            if abs(alpha)<=1:
                mat=[[1,0,0],[0,1,beta],[0,0,1]]
            else:
                mat=[[1,0,-beta*1.0/alpha],[0,1,0],[0,0,1]]
        return mat
    # numpy其实有np.linspace(x1,x2,n)函数...
    def linespace(self,x1,x2,n):
        delta=(x2-x1)*1.0/(n-1)
        print delta
        vector=[]
        for i in range(0,n-1):
            vector.append(x1+delta*i)
        vector.append(x2)
        return vector
    # n是渐变的层数
    def gradation2Color(self,color1,color2,n):
        r1,g1,b1=color1
        r2,g2,b2=color2
        vec1=self.linespace(r1,r2,n)
        vec2=self.linespace(g1,g2,n)
        vec3=self.linespace(b1,b2,n)
        res=[]
        for i in range(n):
            res.append([vec1[i],vec2[i],vec3[i]])
        return res

    def gradationPalette(self,palette,n):
        l=len(palette)
        res=[]
        for i in range(0,l-1):
            v=self.gradation2Color(palette[i],palette[i+1],n)
            if i==0:
                res.append(v[0])
            for j in range(1,n):
                res.append(v[j])
        return res
    def linearFit(self,X,Y):
        z1=np.polyfit(X,Y,1)
        return [z1[0],z1[1]]
    def lab2line(self,colors):
        a=[x[1] for x in colors]
        b=[x[2] for x in colors]
        line=self.linearFit(a,b)
        return line
    def recolor(self):
        line_origin=self.lab2line(self.data_lab)
        gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
        self.gradation_rgb=gradation_rgb
        gradation_lab=[]
        for i in range(len(gradation_rgb)):
            gradation_lab.append(rgb2lab(gradation_rgb[i]))
        line_template=self.lab2line(gradation_lab)
        Rmat=self.getRotateMat(line_origin,line_template)
        Tmat1=self.getTranslateMat(line_origin,True)
        Tmat2=self.getTranslateMat(line_template,False)
        
        self.img_copy = self.img_rgba.copy()
        pix=self.img_copy.load()
        width=self.img_copy.size[0]
        height=self.img_copy.size[1]

        mat_X=[]
        L_save=[]
        for x in range(width):
            print x####################################################################################
            for y in range(height):
                #print pix[x,y]
                R,G,B,A=pix[x,y]  # x is horizontal, y is vertical
                Lab=rgb2lab([R,G,B])
                L_save.append(Lab[0])
                mat_X.append([Lab[1],Lab[2],1])
        X_origin=np.asarray(mat_X)

        X_origin=X_origin.transpose()
        X_map=np.dot(Tmat1,X_origin)
        X_map=np.dot(Rmat,X_map)
        X_map=np.dot(Tmat2,X_map)
        X_map=X_map.transpose()
        # 为什么X_map每一行的第三个是0？不应该是1么
        for x in range(width):
            for y in range(height):
                idx=x*height+y
                Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
                rgb=lab2rgb(Lab)
                pix[x,y]=(rgb[0],rgb[1],rgb[2],pix[x,y][3])
        return self.img_copy
    # y=alpha*x+beta
    # 平移是根据color到line的垂直距离的水平和竖直方向来求得
    def getSingleColorTmat(self,line,color):
        Lab=rgb2lab(color)
        a=Lab[1]
        b=Lab[2]
        alpha=line[0]
        beta=line[1]
        l=abs((alpha*a-b+beta)*1.0/(math.sqrt(alpha*alpha+1)))
        print 'l',l
        angle=np.arctan(abs(alpha))
        print 'line',line
        print 'angle',angle*180/math.pi
        delta_x=l*np.sin(angle)
        delta_y=l*np.cos(angle)
        if alpha>=0:
            b2=alpha*a+beta
            # color在直线下方
            if b2>=b:
                matrix=[[1,0,delta_x],[0,1,-delta_y],[0,0,1]]
            # color在直线上方
            else:
                matrix=[[1,0,-delta_x],[0,1,delta_y],[0,0,1]]
        else:
            b2=alpha*a+beta
            if b2>=b:
                matrix=[[1,0,-delta_x],[0,1,-delta_y],[0,0,1]]
            else:
                matrix=[[1,0,delta_x],[0,1,delta_y],[0,0,1]]
        return matrix
    def getMainColor(self):
        count=np.zeros((self.bin_range,self.bin_range,self.bin_range))
        for i in range(len(self.dataArray)):
            R,G,B,A=self.dataArray[i]
            if A>0:
                ri=R/self.bin_size
                gi=G/self.bin_size
                bi=B/self.bin_size
                count[ri][gi][bi]+=1
        maxcnt=0
        for i in range(self.bin_range):
            for j in range(self.bin_range):
                for k in range(self.bin_range):
                    if i==j and j==k:
                        continue
                    #print (i+0.5)*self.bin_size,(j+0.5)*self.bin_size,(k+0.5)*self.bin_size,count[i][j][k]
                    if count[i][j][k]>maxcnt:
                        mainColor=[(i+0.5)*self.bin_size,(j+0.5)*self.bin_size,(k+0.5)*self.bin_size]
                        maxcnt=count[i][j][k]
                    #if count[i][j][k]>5000:
                    #    print [(i+0.5)*self.bin_size,(j+0.5)*self.bin_size,(k+0.5)*self.bin_size]
        print 'maxcnt',maxcnt
        return mainColor
    # 平移矩阵是根据主色调和color之间的ab差距来求得
    def getSingleColorTmat2(self,color):
        Lab=rgb2lab(color)
        mainColor=self.getMainColor()
        mainLab=rgb2lab(mainColor)
        print 'maincolor',mainColor,mainLab
        print 'palette',color,Lab
        delta_x=Lab[1]-mainLab[1]
        delta_y=Lab[2]-mainLab[2]
        matrix=[[1,0,delta_x],[0,1,delta_y],[0,0,1]]
        return matrix

    def getSingleColorTmat3(self,line,mark):
        mainColor=self.getMainColor()
        mainLab=rgb2lab(mainColor)
        alpha1=line[0]
        beta1=line[1]
        if alpha1!=0:
            alpha2=-1*1.0/alpha1
            beta2=mainLab[2]-alpha2*mainLab[1]
            p_x=1.0*(beta2-beta1)/(alpha1-alpha2)
            p_y=alpha1*p_x+beta1
            if mark==True:
                matrix=[[1,0,-p_x],[0,1,-p_y],[0,0,1]]  # 将line从点p平移到原点
            else:
                matrix=[[1,0,p_x],[0,1,p_y],[0,0,1]]  # 将line从原点平移到点p
        else:
            p_x=mainLab[1]
            p_y= beta1
            if mark==True:
                matrix=[[1,0,-p_x],[0,1,-p_y],[0,0,1]]  # 将line从点p平移到原点
            else:
                matrix=[[1,0,p_x],[0,1,p_y],[0,0,1]]  # 将line从原点平移到点p
        return matrix,[p_x,p_y]

    # 如果palette只有一个颜色，平移是根据该颜色到直线的距离来决定
    def recolor_SingleColor(self):
        line_origin=self.lab2line(self.data_lab)
        #gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
        Tmat=self.getSingleColorTmat(line_origin,self.palette[0])
        print 'Tmat1',Tmat
        #Tmat=self.getSingleColorTmat2(self.palette[0])
        ## 注意由于在main里面是同一个pp，所以应该立即先保存，否则img_recolor2改动后，这个也会随之改动
        self.img_copy = self.img_rgba.copy()
        pix=self.img_copy.load()
        width=self.img_copy.size[0]
        height=self.img_copy.size[1]

        mat_X=[]
        L_save=[]
        for x in range(width):
            #print x####################################################################################
            for y in range(height):
                #print pix[x,y]
                R,G,B,A=pix[x,y]  # x is horizontal, y is vertical
                Lab=rgb2lab([R,G,B])
                L_save.append(Lab[0])
                mat_X.append([Lab[1],Lab[2],1])
        X_origin=np.asarray(mat_X)

        X_origin=X_origin.transpose()
        X_map=np.dot(Tmat,X_origin)
        X_map=X_map.transpose()
        # 为什么X_map每一行的第三个是0？不应该是1么
        for x in range(width):
            for y in range(height):
                idx=x*height+y
                Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
                rgb=lab2rgb(Lab)
                pix[x,y]=(rgb[0],rgb[1],rgb[2],pix[x,y][3])
        return self.img_copy
        # 如果palette只有一个颜色，平移是根据主色调与palette之间的ab差距
    def recolor_SingleColor2(self):
        line_origin=self.lab2line(self.data_lab)
        #gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
        #Tmat=self.getSingleColorTmat(line_origin,self.palette[0])
        Tmat=self.getSingleColorTmat2(self.palette[0])
        print 'Tmat2',Tmat
        self.img_copy = self.img_rgba.copy()
        pix=self.img_copy.load()
        width=self.img_copy.size[0]
        height=self.img_copy.size[1]

        mat_X=[]
        L_save=[]
        for x in range(width):
            #print x####################################################################################
            for y in range(height):
                #print pix[x,y]
                R,G,B,A=pix[x,y]  # x is horizontal, y is vertical
                Lab=rgb2lab([R,G,B])
                L_save.append(Lab[0])
                mat_X.append([Lab[1],Lab[2],1])
        X_origin=np.asarray(mat_X)

        X_origin=X_origin.transpose()
        X_map=np.dot(Tmat,X_origin)
        X_map=X_map.transpose()
        # 为什么X_map每一行的第三个是0？不应该是1么
        for x in range(width):
            for y in range(height):
                idx=x*height+y
                Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
                rgb=lab2rgb(Lab)
                pix[x,y]=(rgb[0],rgb[1],rgb[2],pix[x,y][3])
        return self.img_copy
    # 计算mainColor到line1最近的点p1（即正交的交点），p2为palette的点，对line1进行旋转操作使得与p1,p2垂直
    # （首先得将line1平移到原点，再旋转，再平移回到p1点），再将line1平移到p2
    def recolor_SingleColor3(self):
        line_origin=self.lab2line(self.data_lab)
        #gradation_rgb=self.gradationPalette(self.palette,self.gradation_num)
        #self.gradation_rgb=gradation_rgb
        #gradation_lab=[]
        #for i in range(len(gradation_rgb)):
        #    gradation_lab.append(rgb2lab(gradation_rgb[i]))
        #line_template=self.lab2line(gradation_lab)
        Tmat1,point1=self.getSingleColorTmat3(line_origin,True)
        Tmat2,point1=self.getSingleColorTmat3(line_origin,False)
        Lab=rgb2lab(self.palette[0])
        point2=[Lab[1],Lab[2]]
        if point1[0]!=point2[0]:
            alpha=(point2[1]-point1[1])*1.0/(point2[0]-point1[0])
            if alpha!=0:
                alpha2=-1.0/alpha
                beta2=point1[1]-alpha2*point1[0]
                line2=[alpha2,beta2]
                print 'line2:',line2
                Rmat=self.getRotateMat(line_origin,line2)
            else:
                angle2=0.5*math.pi
                angle1=np.arctan(line_origin[0])
                Rmat=self.getRotateMat_angle(angle1,angle2)
        else:
            angle2=0
            angle1=np.arctan(line_origin[0])
            Rmat=self.getRotateMat_angle(angle1,angle2)
        delta_x=point2[0]-point1[0]
        delta_y=point2[1]-point1[1]
        Tmat3=[[1,0,delta_x],[0,1,delta_y],[0,0,1]]
        print 'line1',line_origin
        print 'mainLab','rgb',self.getMainColor(),' lab',rgb2lab(self.getMainColor())
        print 'point1',point1
        print 'palette','rgb:',self.palette[0], ' lab:',Lab
        print 'Tmat1',Tmat1
        print 'Rmat',Rmat
        print 'Tmat2',Tmat2
        print 'Tmat3',Tmat3

        self.img_copy = self.img_rgba.copy()
        pix=self.img_copy.load()
        width=self.img_copy.size[0]
        height=self.img_copy.size[1]

        mat_X=[]
        L_save=[]
        for x in range(width):
            #print x####################################################################################
            for y in range(height):
                #print pix[x,y]
                R,G,B,A=pix[x,y]  # x is horizontal, y is vertical
                Lab=rgb2lab([R,G,B])
                L_save.append(Lab[0])
                mat_X.append([Lab[1],Lab[2],1])
        X_origin=np.asarray(mat_X)

        X_origin=X_origin.transpose()
        X_map=np.dot(Tmat1,X_origin)
        X_map=np.dot(Rmat,X_map)
        X_map=np.dot(Tmat2,X_map)
        X_map=np.dot(Tmat3,X_map)
        X_map=X_map.transpose()
        # 为什么X_map每一行的第三个是0？不应该是1么
        for x in range(width):
            for y in range(height):
                idx=x*height+y
                Lab=[L_save[idx],int(X_map[idx][0]),int(X_map[idx][1])]
                rgb=lab2rgb(Lab)
                pix[x,y]=(rgb[0],rgb[1],rgb[2],pix[x,y][3])
        return self.img_copy
def linespace(x1,x2,n):
    delta=(x2-x1)*1.0/(n-1)
    print delta
    vector=[]
    for i in range(0,n-1):
        vector.append(x1+delta*i)
    vector.append(x2)
    return vector
def save_theme(filename,colors,width,height):
    img=Image.new('RGB',[width,height])
    img_pix=img.load()
    width=img.size[0]
    height=img.size[1]
    k=len(colors)
    xlen=width/k
    for i in range(0,k):
        for x in range(xlen*i,min(xlen*i+xlen,width)):
            for y in range(height):
                R=int(colors[i][0])
                G=int(colors[i][1])
                B=int(colors[i][2])
                img_pix[x,y]=(R,G,B)
    # 剩余的就补最后一个颜色
    for x in range(xlen*(k-1)+xlen,width):
        for y in range(height):
            R=int(colors[k-1][0])
            G=int(colors[k-1][1])
            B=int(colors[k-1][2])
            img_pix[x,y]=(R,G,B)
    img.save(filename)