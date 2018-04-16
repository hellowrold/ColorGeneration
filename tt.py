from PIL import Image, ImageDraw, ImageOps
import numpy as np 
import math
from color2 import *
import hsl
def  img2data(filename):
	img = Image.open(filename)
	#img.show()
	#if img.mode != 'RGB':
	#	img = img.convert('RGB')

	#img_dataArray = np.asarray(img)
	img_rgb=img.convert('RGB')
	#img2=img.copy()
	#print img_dataArray
	#data1=list(img.getdata())
	#data2 = list(img2.getdata())
	#print list(img2.getdata())

	#print data1
	#print data2
	#print data2[1][0],data2[1][1],data2[1][2]
	#out_array=data2
	#out_array[1]=(100,99,98)
	#data2[1]=(100,99,98)
	#print 'data2'
	#print data2[1][0],data2[1][1],data2[1][2]
	#data3=img2.getdata();
	#print 'data3'
	#print data3[1][0],data3[1][1],data3[1][2]
	#pix=img2.load()
	pix=img.load()
	pix_rgb=img_rgb.load()
	print pix[0,0],pix[1,0],pix[2,0],pix[3,0]
	print pix_rgb[0,0],pix_rgb[1,0],pix_rgb[2,0],pix_rgb[3,0]
	pix_rgb[0,0]=(100,99,-11)
	print pix[0,0]
	print pix_rgb[0,0]

	#img2.save('res.png')
	#print list(img2.getdata())
def test(list1):
	list1[1][0]=255

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
	img.save(filename)
def save_gration(filename,colors,width,height):
	k=len(colors)
	img=Image.new('RGB',[width*k,height])
	img_pix=img.load()
	width=img.size[0]
	height=img.size[1]
	
	for i in range(0,k):
		r,g,b=colors[i]
		h,s,l=hsl.rgb2hsl([r,g,b])

	for i in range(0,k):
		for x in range(xlen*i,min(xlen*i+xlen,width)):
			for y in range(height):
				R=int(colors[i][0])
				G=int(colors[i][1])
				B=int(colors[i][2])
				img_pix[x,y]=(R,G,B)
	img.save(filename)

if __name__ == '__main__':
	#img2data('input/tt.png')
	
	'''
	n = 2
	m = 3
	k = 4
	matrix = [None]*2
	print '1',matrix
	
	for i in range(len(matrix)):
		matrix[i] = [0]*3
	print(matrix)

	for i in range(n):
		for j in range(m):
			matrix[i][j] = [1]*k
    
	print(matrix)
	matrix[0][0][0]=[2,2,3]
	print matrix
	mat=np.zeros((2,2,3))
	print mat
	mat[0][0]=[1,2,3]
	mat[0][1]=[4,5,6]
	print mat
	a,b,c=[1,2,3]
	print 'abc',a,b,c
	mat=[[1,2,3],[2,3,4]]
	l=len(mat)
	print 'lenl',l
	'''
	#print lab

	lab=[0,100,-100]
	print lab2rgb(lab)
	lab=[1,-0.7,1]
	print lab2rgb(lab)
