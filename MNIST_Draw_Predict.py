###########################################################################
#
# Handwritten digit recognition
# 
#
# Author: Sam Showalter
# Date: November 1, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#System based imports
import sys
import time

#Animation and visualization imports
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageGrab, Image, ImageDraw, ImageTk
import PIL.ImageOps
import cv2

#Scientific Library imports
import numpy as np
from scipy.ndimage.measurements import center_of_mass

#Deep learning imports
import tensorflow as tf 
import keras

###########################################################################
# Class and Constructor
###########################################################################


model = keras.models.load_model('C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\BA_Source_Code\\Neural_Networks\\Convolutional_Nets\\MNIST_conv_net.h5')





class DigitRecognition( Frame ):
	def __init__(self,
				 model_dir = 'C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\BA_Source_Code\\Neural_Networks\\Convolutional_Nets\\MNIST_conv_net.h5'):
		
		#Initialize the tkinter frame and pack
		Frame.__init__( self )
		self.pack( expand = YES, fill = BOTH )

		#Load the model
		self.model = keras.models.load_model(model_dir)

		#Set a title and geometry for the package
		#TODO: Set to make it exactly half of the screen, or whole potentially
		self.master.title( "MNIST draw and predict" )
		self.master.geometry( "1000x800" )

		#Create new image to be drawn on
		self.img = Image.new("L", (280,280), "white")
		self.draw = ImageDraw.Draw(self.img)

		# create Canvas component, where drawing will happen
		self.myCanvas = Canvas( self, highlightthickness=2, highlightbackground="black",width = 280, height = 280 )
		self.myCanvas.pack( side = LEFT, expand = NO, fill = NONE )
		self.myCanvas.place(x= 50,y = 50)

		self.inputCanvas = Canvas( self, highlightthickness=2, highlightbackground="black",width = 280, height = 280 )
		self.inputCanvas.pack( side = LEFT, expand = NO, fill = NONE )
		self.inputCanvas.place(x = 50, y = 400)

		#Set output box where predictions will go
		self.outputCanvas = Canvas( self, highlightthickness=2, highlightbackground="black",width = 75, height = 75 )
		self.outputCanvas.pack( side = LEFT, expand = NO, fill = NONE )
		self.outputCanvas.place(x = 500, y = 100)

		#Probs canvas
		self.probsCanvas = Canvas( self, highlightthickness=2, highlightbackground="black",width = 400, height = 210 )
		self.probsCanvas.pack( side = LEFT, expand = NO, fill = NONE )
		self.probsCanvas.place(x = 450, y = 250)
		self.draw_probabilities()
		
		
		#Set clear and recognition button
		clear_button = Button(self, text = "Clear", command = self.clear_canvas)
		clear_button.place(x = 125, y = 340)
		recognize_button = Button(self, text = "Recognize", command = self.recognize_drawing)
		recognize_button.place(x = 190, y = 340)
      
		#Bind mouse dragging event to the canvas
		self.myCanvas.bind( "<B1-Motion>", self.paint_digit)


	def paint_digit( self, event ):
		#Get coordinates for making the circle
		x1, y1 = ( event.x - 10), ( event.y - 10)
		x2, y2 = ( event.x + 10), ( event.y + 10)

		#Draw on hidden PIL object
		self.draw.ellipse([x1, y1, x2, y2], fill = 'black')

		#Draw on canvas that you can see in tkinter
		self.myCanvas.create_oval( x1, y1, x2, y2, fill = "black" )

	def center_shift_define(self, img):

		#Get y and x coordinates for center of mass
		cy, cx = center_of_mass(img)

		#Get dimensions of the image, and set the shift
		rows, cols = img.shape
		shiftx = np.round(cols / 2.0 - cx).astype(int)
		shifty = np.round(rows / 2.0 - cy).astype(int)

		#Return the x and y shift
		return shiftx, shifty 

	def shift_picture(self, img, sx, sy):

		#Get the shape of the image
	    rows, cols = img.shape

	    #Matrix for affine warp, includes appropriate x and y shift
	    M = np.float32([[1, 0, sx], [0, 1, sy]])

	    #Create a new image using warp affine command, and return it
	    shifted = cv2.warpAffine(img, M, (cols, rows))
	    return shifted

	def clear_canvas(self):

		#Clear my canvas and the output canvas
		self.myCanvas.delete('all')
		self.outputCanvas.delete('all')
		self.inputCanvas.delete('all')
		self.probsCanvas.delete('probs')

		#Clear the pillow image behind the output canvas
		self.img = Image.new("L", (280,280), "white")
		self.draw = ImageDraw.Draw(self.img)
    
	def reshape_helper(self, gray, rows, cols, target_size = 20):

		#If rows in cropped image are longer than cols
		if rows > cols:
			#Reduction factor
		    factor = 20.0/rows
		    rows = target_size
		    cols = int(round(cols*factor))
		    gray = cv2.resize(gray, (cols,rows))

		#If columns are greater than or equal to the rows
		else:
			#Reduction factor
		    factor = 20.0/cols
		    cols = target_size
		    rows = int(round(rows*factor))
		    gray = cv2.resize(gray, (cols, rows))

		#Add padding to this to make image centered
		colsPadding = (int(np.ceil((28-cols)/2.0)),int(np.floor((28-cols)/2.0)))
		rowsPadding = (int(np.ceil((28-rows)/2.0)),int(np.floor((28-rows)/2.0)))

		#Add padding, and set image
		gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

		self.final_img = gray

	def reshape_pic(self, img):

		#Take copy of image
		img = np.array(img)

		#Top
		while np.sum(img[0]) == 0:
		    img = img[1:]

		#Left
		while np.sum(img[:,0]) == 0:
		    img = np.delete(img,0,1)

		#Bottom
		while np.sum(img[-1]) == 0:
		    img = img[:-1]

		#Right
		while np.sum(img[:,-1]) == 0:
		    img = np.delete(img,-1,1)

		#New row and column information
		rows,cols = img.shape

		#Help reshape the model
		self.reshape_helper(img, rows, cols)

	def drawing_preprocessing(self):
		self.final_img = 255 - np.array(self.img)
		self.reshape_pic(self.final_img)
		sx, sy = self.center_shift_define(np.array(self.final_img))
		self.final_img = self.shift_picture(np.array(self.final_img), sx, sy)
		self.final_image = cv2.resize(self.final_img, (28, 28), interpolation=cv2.INTER_AREA)
		self.viz_image = self.final_image
		self.final_image = np.array(self.final_image).reshape(1,28,28,1) / 255.0

	def draw_system_input(self):
		self.viz_image = cv2.resize(self.viz_image, (280, 280), interpolation=cv2.INTER_AREA)
		self.viz_image=Image.frombytes('L', (self.viz_image.shape[1],self.viz_image.shape[0]), self.viz_image.astype('b').tostring())
		self.viz_image = ImageTk.PhotoImage(image=self.viz_image)
		self.inputCanvas.create_image(2,2,anchor='nw',image=self.viz_image)

	def draw_prob_bars(self):
		for j in range(10):
				fillc = 'blue'
				if j == self.pred:
					fillc = 'green'

				self.probsCanvas.create_rectangle(110 ,10 + j*20,110 + self.probs[j]*2,10 + j*20 + 10,tags = "probs",  fill=fillc)
				self.probsCanvas.create_text(350,15 + j*20,fill="darkblue",font="Arial 10",anchor = 'w',tags = 'probs', text= "{} %".format(str(self.probs[j])))




	def draw_probabilities(self):
		for i in range(10):
			self.probsCanvas.create_text(15,15 + i*20,fill="darkblue",font="Arial 10",anchor = 'w',text= "Probability of %s:"%(str(i)))


	def recognize_drawing(self):

		# No matter what, clear out the previous images for prediction and viz
		self.inputCanvas.delete('all')
		self.outputCanvas.delete('all')
		self.probsCanvas.delete('probs')
		
		#Run all drawing preprocessing
		self.drawing_preprocessing()

		#Draw the system input so the user can see
		self.draw_system_input()

		#Predict the result and give the output
		self.probs = model.predict(self.final_image)*100
		self.probs = np.squeeze(self.probs.astype(int))
		self.pred = int(model.predict_classes(self.final_image))
		self.draw_prob_bars()

		
		self.outputCanvas.create_text(38,38,fill="darkblue",font="Times 30 bold",text=str(self.pred))

      
def main():
   DigitRecognition().mainloop()


if __name__ == "__main__":
   main()