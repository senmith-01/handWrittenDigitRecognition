import pygame,sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX= 640
WINDOWSIZEY= 480

#initialize the pygame
pygame.init()

pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))