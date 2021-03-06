#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:01:36 2018

@author: sylvain
"""

import matplotlib
matplotlib.use('Agg')

from models import model
import VQALoader
import VocabEncoder
import torchvision.transforms as T
import torch
from torch.autograd import Variable

import numpy as np
import json
import matplotlib.pyplot as plt

import pickle
import os
import datetime
# from shutil import copyfilefrom
from rich.progress import track
from tqdm import tqdm


def train(network, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, Dataset='HR'):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()#weight=weights)
   
    best_loss = 10000000
    trainLoss = []
    valLoss = []
    if Dataset == 'HR':
        accPerQuestionType = {'area': [], 'presence': [], 'count': [], 'comp': []}
    else:
        accPerQuestionType = {'rural_urban': [], 'presence': [], 'count': [], 'comp': []}
    OA = []
    AA = []
    for epoch in range(num_epochs):
        
         
        if Dataset == 'HR':
            countQuestionType_train = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            rightAnswerByQuestionType_train = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
        else:
            countQuestionType_train = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
            rightAnswerByQuestionType_train = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
        network.train()
        runningLoss = 0
        for i, data in enumerate(tqdm(train_loader)):
            question, answer, image, type_str, question_str = data
            question = Variable(question.long()).cuda()
            answer = Variable(answer.long()).cuda().resize_(question.shape[0])
            image = Variable(image.float()).cuda()
            
            pred = network(image, question, question_str)
            loss = criterion(pred, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.cpu().item() * question.shape[0]

            answer = answer.cpu().numpy()
            pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
            for j in range(answer.shape[0]):
                countQuestionType_train[type_str[j]] += 1
                if answer[j] == pred[j]:
                    rightAnswerByQuestionType_train[type_str[j]] += 1
            
        loss = runningLoss / len(train_dataset)
        trainLoss.append(loss)
        if loss < best_loss:
            best_loss = loss
            state = {
                'net': network.state_dict(),
                'loss': loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')

        print('epoch %d loss: %.3f' % (epoch, trainLoss[epoch]))
        print('val accuracies:')
        for i, key in enumerate(rightAnswerByQuestionType_train):
            print(key + ": " + str(100 * rightAnswerByQuestionType_train[key] / countQuestionType_train[key]) + "%")

        with torch.no_grad():
            network.eval()
            runningLoss = 0
            if Dataset == 'HR':
                countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
                rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            else:
                countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
                rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
            count_q = 0
            for i, data in enumerate(tqdm(validate_loader)):
                question, answer, image, type_str, question_str, image_original = data
                question = Variable(question.long()).cuda()
                answer = Variable(answer.long()).cuda().resize_(question.shape[0])
                image = Variable(image.float()).cuda()

                pred = network(image,question, question_str)
                loss = criterion(pred, answer)
                runningLoss += loss.cpu().item() * question.shape[0]
                
                answer = answer.cpu().numpy()
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1
    
            valLoss.append(runningLoss / len(validate_dataset))
            print('epoch %d, val loss: %.3f' % (epoch, valLoss[epoch]))
            print('val accuracies:')
            for i, key in enumerate(rightAnswerByQuestionType):
                print(key + ": " + str(100*rightAnswerByQuestionType[key]/countQuestionType[key]) + "%")
        
            numQuestions = 0
            numRightQuestions = 0
            currentAA = 0
            for type_str in countQuestionType.keys():
                if countQuestionType[type_str] > 0:
                    accPerQuestionType[type_str].append(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
                numQuestions += countQuestionType[type_str]
                numRightQuestions += rightAnswerByQuestionType[type_str]
                currentAA += accPerQuestionType[type_str][epoch]
            accuracy = numRightQuestions *1.0 / numQuestions 
            print("total val accuracy: %.2f%%" % (accuracy*100))
            OA.append(accuracy)
            AA.append(currentAA * 1.0 / 4)
        e_max = np.argmax(OA)
        print("max accuracy is %.2f%% at epoch %d" % (OA[e_max]*100, e_max))
    return trainLoss, valLoss, accPerQuestionType  
                

if __name__ == '__main__':
    disable_log = True
    batch_size = 8
    num_epochs = 150
    learning_rate = 0.00001
    ratio_images_to_use = 0.01
    Dataset = 'HR'
    load_checkpoint = False

    if Dataset == 'LR':
        data_path = '/media/enbozhou/Data/EnboZhou/VQA_data/RSVQA_LR/'
        allquestionsJSON = os.path.join(data_path, 'questions.json')
        allanswersJSON = os.path.join(data_path, 'answers.json')
        questionsJSON = os.path.join(data_path, 'LR_split_train_questions.json')
        answersJSON = os.path.join(data_path, 'LR_split_train_answers.json')
        imagesJSON = os.path.join(data_path, 'LR_split_train_images.json')
        questionsvalJSON = os.path.join(data_path, 'LR_split_val_questions.json')
        answersvalJSON = os.path.join(data_path, 'LR_split_val_answers.json')
        imagesvalJSON = os.path.join(data_path, 'LR_split_val_images.json')
        images_path = os.path.join(data_path, 'Images_LR/')
    else:
        data_path = '/media/enbozhou/Data/EnboZhou/VQA_data/RSVQA_HR/'
        images_path = os.path.join(data_path, 'Data/')
        allquestionsJSON = os.path.join(data_path, 'USGSquestions.json')
        allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
        questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
        answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
        imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')
        questionsvalJSON = os.path.join(data_path, 'USGS_split_val_questions.json')
        answersvalJSON = os.path.join(data_path, 'USGS_split_val_answers.json')
        imagesvalJSON = os.path.join(data_path, 'USGS_split_val_images.json')
    encoder_questions = VocabEncoder.VocabEncoder(allquestionsJSON, questions=True)
    if Dataset == "LR":
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = True)
    else:
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = False)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),            
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    
    if Dataset == 'LR':
        patch_size = 256
    else:
        patch_size = 512   
    train_dataset = VQALoader.VQALoader(images_path, imagesJSON, questionsJSON, answersJSON, encoder_questions, encoder_answers, train=True, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
    validate_dataset = VQALoader.VQALoader(images_path, imagesvalJSON, questionsvalJSON, answersvalJSON, encoder_questions, encoder_answers, train=False, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size)
    
    RSVQA = model.VQAModel(encoder_questions.getVocab(), encoder_answers.getVocab(), input_size = patch_size).cuda()
    if load_checkpoint:
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        RSVQA.load_state_dict(checkpoint['net'])
    trainLoss, valLoss, accPerQuestionType = train(RSVQA, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, Dataset)
    

    trainLoss = np.array(trainLoss) 
    np.savetxt('trainLoss_ds_fine.csv', trainLoss, delimiter=',')
    valLoss = np.array(valLoss) 
    np.savetxt('valLoss_ds_fine.csv', valLoss, delimiter=',')

    with open('accPerQuestionType_ds_fine.json', 'w') as fileObject:
        jsObj = json.dumps(accPerQuestionType)
        fileObject.write(jsObj)