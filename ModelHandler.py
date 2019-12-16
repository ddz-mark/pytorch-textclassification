# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd


class ModelHandler(nn.Module):
    def __init__(self, params):
        super(ModelHandler, self).__init__()
        self.device = params['device']
        self.epoch_num = params['epoch_num']
        self.train_batch_size = params['train_batch_size']
        self.val_batch_size = params['val_batch_size']
        self.model = params['model'].to(self.device)

    def forward(self, x):
        return self.model(x)

    def fit(self, train_iter, loss_fn, optimizer, modelPath=None,
            val_iter=None, early_stopping_rounds=None, verbose=2):
        '''
        early_stopping_rounds：是否使用 early_stop 来防止过拟合
        verbose：2分类还是多分类
        '''

        if early_stopping_rounds != None:
            # 用来计数多少epoch在验证集上的结果没有改进了
            count = 0

        batch_size = self.train_batch_size
        batchNumInEveryEpoch = len(train_iter)

        epochNums = self.epoch_num
        best_val_acc = -1000000
        best_val_loss = 1000000

        num = 0
        for epoch in range(epochNums):
            print('************************* epoch:', epoch, '*************************')

            self.train()
            torch.set_grad_enabled(True)

            # 随机化数据
            #             randPermNums = np.random.permutation(trainDataNum)
            #             X_train = X_train[randPermNums]
            #             y_train = np.array(y_train)[randPermNums]

            trainAcc = 0.0
            trainLoss = 0.0
            for index, (X_train_var, y_train_var) in enumerate(train_iter):
                X_train_var = (i.to(self.device) for i in X_train_var)
                
                # 将 label 转为 one-hot编码，这里针对多分类和二分类的softmax形式。如果是二分类的sogmid，则注释
                y_train_var = y_train_var.unsqueeze(1)
                y_train_var = torch.zeros(len(y_train_var), verbose).scatter_(1, y_train_var, 1)
                y_train_var = y_train_var.to(self.device)

                self.zero_grad()
                #                 print (X_train_var.shape)
                scores = self.forward(X_train_var)
                loss = loss_fn(scores.squeeze(), y_train_var)
                trainAcc = trainAcc + self.getAUC(y_train_var, torch.sigmoid(scores).squeeze())
                trainLoss = trainLoss + loss.data.item()
                self.train()
                torch.set_grad_enabled(True)
                loss.backward()
                optimizer.step()
            if verbose == 2:
                print('train auc:', trainAcc / float(batchNumInEveryEpoch))
                print('train loss:', trainLoss / float(batchNumInEveryEpoch))

            val_acc, val_loss = self.check_accuracy(self, val_iter, self.val_batch_size, loss_fn,
                                                    True, verbose)
            if verbose == 2:
                print('val_auc:', val_acc)
                print('val_loss:', val_loss)
            #     print ('val_acc:', val_acc, file=file, flush=True)
            if val_acc > best_val_acc:
                #             if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                bestEpoch = epoch
                count = 0
                if modelPath != None:
                    torch.save(self.state_dict(), modelPath)
            elif early_stopping_rounds != None:
                count += 1
                if count >= early_stopping_rounds:
                    if verbose >= 1:
                        print('Stopping.')
                        print('Best Epoch:', bestEpoch)
                        print('Best Val Auc:', best_val_acc)
                        print('Best Val Loss:', best_val_loss)
                    break

    def check_accuracy(self, model, val_loader, valBatchSize, loss_fn,
                       isTrain, verbose, temperature=1):
        if verbose == 2:
            if isTrain:
                print('*****Checking accuracy on validation set*****')
            #         print('Checking accuracy on validation set', file=file, flush=True)
            else:
                print('Checking accuracy on test set')
                #         print('Checking accuracy on test set', file=file, flush=True)
        self.eval()
        torch.set_grad_enabled(False)

        batchNum = len(val_loader)
        #         batchNum = X_val.shape[0] // valBatchSize
        #         if isTrain != True and X_val.shape[0] % valBatchSize != 0:
        #             batchNum += 1
        if verbose == 2:
            print('batchNum:', batchNum)

        valAcc = 0.0
        valLoss = 0.0
        for index, (tX_val_var, tY_val_var) in enumerate(val_loader):
            tX_val_var = (i.to(self.device) for i in tX_val_var)

            tY_val_var = tY_val_var.unsqueeze(1)
            tY_val_var = torch.zeros(len(tY_val_var), verbose).scatter_(1, tY_val_var, 1)
            tY_val_var = tY_val_var.to(self.device)

            scores = self.forward(tX_val_var)

            if isTrain == True:
                loss = loss_fn(scores.squeeze(), tY_val_var / temperature)
                valAcc += self.getAUC(tY_val_var / temperature, torch.sigmoid(scores).squeeze())
                valLoss = valLoss + loss.data.item()

        if isTrain == True:
            return valAcc / float(batchNum), valLoss / float(batchNum)

    def predict_proba(self, testDF, inputType='tensor', temperature=1):
        self.eval()
        torch.set_grad_enabled(False)
        if inputType == 'tensor':
            testDF = testDF.reshape(testDF.shape[0], 1, -1).to(self.device)
            scores = self.forward(testDF) / temperature
            return torch.sigmoid(scores).squeeze()
        elif inputType == 'DataFrame':
            testDF = testDF.reshape(testDF.shape[0], 1, -1).to(self.device)
            return torch.sigmoid(self.forward(
                torch.tensor(np.array(testDF), dtype=torch.float32, device=self.device)) / temperature).squeeze()
        torch.set_grad_enabled(True)

    def predict(self, testDF, inputType='tensor', threshold=0.5, temperature=1):
        predict_proba = self.predict_proba(testDF, inputType, temperature).cpu().numpy().tolist()
        #         predict_lables = [1 if x >= threshold else 0 for x in predict_proba]
        return predict_proba

    def getAUC(self, y_true, y_score):
        return roc_auc_score(y_true.detach().cpu().numpy(),
                             y_score.detach().cpu().numpy())

    # In[ ]:

# 老版 ModelHandler，主要区别在于数据没有经过 iter。
# class ModelHandler(nn.Module):
#     def __init__(self, params):
#         super(ModelHandler, self).__init__()
# #         self.bestStateDict = None
#         self.epochNums = params['epochNums']
#         self.batch_size = params['batch_size']
#         self.device = params['device']
#         self.dnn = params['model'].to(self.device)


#     def forword(self, features):
#         return self.dnn(features)

# #     def reset(self, m):
# #         if hasattr(m, 'reset_parameters'):
# #             torch.cuda.manual_seed(1)
# #             m.reset_parameters()


#     def fit(self, X_train, y_train, loss_fn, optimizer, task, device, modelPath=None,
#             eval_set=None, early_stopping_rounds=None, valBatchSize=None, 
#             verbose=0, temperature=1):

#         if eval_set != None:
#             X_val, y_val = eval_set
#         if early_stopping_rounds != None:
#             # 用来计数多少epoch在验证集上的结果没有改进了
#             count = 0
# #         self.apply(self.reset)

#         batch_size = self.batch_size
#         trainDataNum = X_train["feature_idx"].shape[0]
#         batchNumInEveryEpoch = trainDataNum // batch_size
#         epochNums = self.epochNums
#         best_val_acc = -1000000
#         best_val_loss = 1000000
#         if valBatchSize != None:
#             valBatchSize = valBatchSize
#         else:
#             valBatchSize = X_val["feature_idx"].shape[0]
#         num = 0
#         for epoch in range(epochNums):
#             print ('epoch:', epoch)
#         #     print ('epoch:', epoch, file=file, flush=True)
#             # 设置成 training 模式
#             self.train()
#             # 设置自动微分
#             torch.set_grad_enabled(True)

# #             randPermNums = torch.randperm(trainDataNum)
# #             X_train["feature_idx"] = X_train["feature_idx"][randPermNums]
# #             X_train["feature_values"] = X_train["feature_values"][randPermNums]
# #             y_train = y_train[randPermNums]  

#             randPermNums = np.random.permutation(trainDataNum)
#             X_train["feature_idx"] = X_train["feature_idx"].iloc[randPermNums]
#             X_train["feature_values"] = X_train["feature_values"].iloc[randPermNums]
#             y_train = y_train[randPermNums]  


#             trainAcc = 0.0
#             trainLoss = 0.0
#             for t1 in range(batchNumInEveryEpoch):
#                 X_train_var = {}
#                 X_train_var["feature_idx"] = X_train["feature_idx"][t1 * batch_size:(t1 + 1) * batch_size]
#                 X_train_var["feature_values"] = X_train["feature_values"][t1 * batch_size:(t1 + 1) * batch_size]
#                 y_train_var = y_train[t1 * batch_size:(t1 + 1) * batch_size].to(self.device)
#                 self.zero_grad()
#                 scores = self.forword(X_train_var)
#                 loss = loss_fn(scores.squeeze(), y_train_var)
#                 trainAcc = trainAcc + self.getAUC(y_train_var, torch.sigmoid(scores).squeeze())
#                 trainLoss = trainLoss + loss.sst2_data.item()
#                 self.train()
#                 torch.set_grad_enabled(True)
#                 loss.backward()
#                 optimizer.step()
#             if verbose == 2:
#                 print ('train acc:', trainAcc / float(batchNumInEveryEpoch))
#                 print ('train loss:', trainLoss / float(batchNumInEveryEpoch))

#             val_acc, val_loss = self.check_accuracy(self, X_val, y_val, valBatchSize, loss_fn, 
#                                                     task, device, True, verbose)   
#             if verbose == 2:
#                 print ('val_acc:', val_acc)
#                 print ('val_loss:', val_loss)
#         #     print ('val_acc:', val_acc, file=file, flush=True)
#             if val_acc > best_val_acc:
# #             if val_loss < best_val_loss:
#                 best_val_acc = val_acc
#                 best_val_loss = val_loss
#                 bestEpoch = epoch
#                 count = 0
#                 if modelPath != None:
#                     torch.save(self.state_dict(), modelPath)
#             elif early_stopping_rounds != None:
#                 count += 1
#                 if count >= early_stopping_rounds:
#                     if verbose >= 1:
#                         print ('Stopping.')
#                         print ('Best Epoch:', bestEpoch)
#                         print ('Best Val Acc:', best_val_acc)
#                         print ('Best Val Loss:', best_val_loss)
#                     break

#     def check_accuracy(self, model, X_val, y_val, valBatchSize, loss_fn, 
#                        task, device, isTrain, verbose, temperature=1):
#         if verbose == 2:
#             if isTrain:
#                 print('*****Checking accuracy on validation set*****')
#         #         print('Checking accuracy on validation set', file=file, flush=True)
#             else:
#                 print('Checking accuracy on test set') 
#         #         print('Checking accuracy on test set', file=file, flush=True) 
#         # 将模型设置成evaluation模式
#         self.eval()
#         torch.set_grad_enabled(False)
#         batchNum = X_val["feature_idx"].shape[0] // valBatchSize 
#         if isTrain != True and X_val["feature_idx"].shape[0] % valBatchSize != 0:
#             batchNum += 1  
#         if verbose == 2:
#             print ('batchNum:', batchNum)
#         valAcc = 0.0
#         valLoss = 0.0
#         for t1 in range(batchNum): 
#             if isTrain != True and t1 == batchNum - 1:
#                 tX_val_var = X_val[t1 * valBatchSize:]
#                 tY_val_var = y_val[t1 * valBatchSize:].to(self.device)
#             else:
#                 tX_val_var = {}
#                 tX_val_var["feature_idx"] = (X_val["feature_idx"][t1 * valBatchSize:(t1 + 1) * valBatchSize])
#                 tX_val_var["feature_values"] = (X_val["feature_values"][t1 * valBatchSize:(t1 + 1) * valBatchSize])
#                 tY_val_var = y_val[t1 * valBatchSize:(t1 + 1) * valBatchSize].to(self.device)

#             scores = self.forword(tX_val_var)

#             if isTrain == True:
#                 loss = loss_fn(scores.squeeze(), tY_val_var / temperature)
#                 valAcc += self.getAUC(tY_val_var / temperature, torch.sigmoid(scores).squeeze())
#                 valLoss = valLoss + loss.sst2_data.item()

#         if isTrain == True:
#             return valAcc / float(batchNum), valLoss / float(batchNum)

#     def predict_proba(self, testDF, inputType='tensor', temperature=1):
#         self.eval()
#         torch.set_grad_enabled(False)
#         if inputType == 'tensor':
#             scores = self.forword(testDF) / temperature
#             return torch.sigmoid(scores).squeeze()
#         elif inputType == 'DataFrame':
#             return torch.sigmoid(self.forword(torch.tensor(np.array(testDF), dtype=torch.float32, device=device)) / temperature)
#         torch.set_grad_enabled(True)

#     # 改进地方
#     def predict(self, testDF, inputType='tensor', temperature=1, threshold=0.5):
#         predict_proba = self.predict_proba(testDF, inputType, temperature).cpu().numpy().tolist()
#         return predict_proba

#     def getAUC(self, y_true, y_score):
#         return roc_auc_score(y_true.detach().cpu().numpy(), 
#                              y_score.detach().cpu().numpy())  


# In[ ]:
