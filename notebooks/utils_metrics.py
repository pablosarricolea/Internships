import numpy as np
from tensorflow.keras import backend as K

def dice_single(true,pred):
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = K.round(pred)

    intersection = K.sum(true * pred, axis=-1)
    true = K.sum(true, axis=-1)
    pred = K.sum(pred, axis=-1)

    return ((2*intersection) + K.epsilon()) / (true + pred + K.epsilon())

def dice_inner_0(true,pred,index=0):

    #get only the desired class
    true = true[:,:,:,index]
    pred = pred[:,:,:,index]

    #return dice per class
    return dice_single(true,pred)

def dice_inner_1(true,pred,index=1):

    #get only the desired class
    true = true[:,:,:,index]
    pred = pred[:,:,:,index]

    #return dice per class
    return dice_single(true,pred)

def dice_inner_2(true,pred,index=2):

    #get only the desired class
    true = true[:,:,:,index]
    pred = pred[:,:,:,index]

    #return dice per class
    return dice_single(true,pred)

def dice_inner_3(true,pred,index=3):

    #get only the desired class
    true = true[:,:,:,index]
    pred = pred[:,:,:,index]

    #return dice per class
    return dice_single(true,pred)

def dice_inner_4(true,pred,index=4):

    #get only the desired class
    true = true[:,:,:,index]
    pred = pred[:,:,:,index]

    #return dice per class
    return dice_single(true,pred)

def DiceScore(individual_dices):
    #individual_dices: array containing dice scores for each class
    return np.mean(individual_dices)

def Sensitivity(TP, FN):
    return TP / (TP + FN)

def Specificity(TN, FP):
    return TN / (FP + TN)