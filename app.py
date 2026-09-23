import pygame,sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX= 640
WINDOWSIZEY= 480

BOUNDARYINC=5
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)

IMAGESAVE = False

MODEL=load_model('C:\\Users\\ranid\\Downloads\\AI_Project\\best_model.h5')

LABELS={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}

#initialize the pygame
pygame.init()

#FONT=pygame.font.Font("freesansbold.ttf",18)
DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))

isWriting=False

number_xcord=[]
number_ycord=[]

imageCount=1
PREDICT=True

while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()

        if event.type==MOUSEMOTION and isWriting:
            xcord,ycord=event.pos    
            pygame.draw.circle(DISPLAYSURF,WHITE, (xcord,ycord), 4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type==MOUSEBUTTONDOWN:
            isWriting=True    

        if event.type==MOUSEBUTTONUP:
            isWriting=False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)

            rectMinX,rectMaxX=max(number_xcord[0]-BOUNDARYINC,0),min(WINDOWSIZEX,number_xcord[-1]+BOUNDARYINC)
            rectMinY,rectMaxY=max(number_ycord[0]-BOUNDARYINC,0),min(WINDOWSIZEY,number_ycord[-1]+BOUNDARYINC)

            number_xcord=[]
            number_ycord=[]

            img_arr=np.array(pygame.PixelArray(DISPLAYSURF))[rectMinX:rectMaxX,rectMinY:rectMaxY].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png",img_arr) 
                imageCount+=1

            if PREDICT:
                image=cv2.resize(img_arr,(28,28))
                image=np.pad(image,(10,10),'constant',constant_values=0)    
                image=cv2.resize(image,(28,28))/255 

                label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurface=pygame.font.Font(None, 32).render(label,True,RED,WHITE)
                testRectObj=textSurface.get_rect()
                testRectObj.left,testRectObj.bottom=rectMinX,rectMaxY

                DISPLAYSURF.blit(textSurface,testRectObj)

            if event.type==KEYDOWN:
                if event.unicode== 'n':
                    DISPLAYSURF.fill(BLACK)    

    pygame.display.update()                