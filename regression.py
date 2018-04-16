# -*- coding: utf-8 -*-
import numpy as np
def linearFit(X,Y):
	z1=np.polyfit(X,Y,1)
	return [z1[0],z1[1]]

#X=[ 1 ,2  ,3 ,4 ,5 ,6]
#Y=[ 2.5 ,4 ,7 ,8.5 ,10.3 ,12.7]
#cof=linearFit(X,Y)
#p1=np.poly1d(cof)
#print cof  #[ 1.          1.49333333]
#print p1  # 1 x + 1.493