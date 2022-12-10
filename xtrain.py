import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import os
import sys
import signal
import filetype
import pandas as pd
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import repeat
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from imblearn.metrics import specificity_score
from torchvision.transforms.functional import resize
from CoAtnet.coatnet_pytorch_master.coatnet import *
from Swin_Transformer.Swin_Transformer_main.models.swin_transformer import SwinTransformer
from ConvMixer.convmixer_main.convmixer import *
from ViG.Efficient_AI_Backbones_master.vig_pytorch.pyramid_vig import *
from Refiner_ViT.Refiner_ViT_master.models.refined_transformer import *
from Dvit.dvit_repo_master.models.deep_vision_transformer import *
from CVT_Refiner.vit_pytorch_main.vit_pytorch.cvt import CvT

# 訓練資料路徑
train_data_path = r'D:\Scalogram\洪健軒\data\train'
# 驗證資料路徑
val_data_path = r"D:\Scalogram\洪健軒\data\valid"

dir = r'D:\Scalogram\洪健軒\weight\train_and_test_itself\ConvMixer'
weight_path = os.path.join(dir, 'epoch_')
txt_path = os.path.join(dir, 'result.txt')
cf_path = os.path.join(dir, 'confusion_matrix_')
npy_path = os.path.join(dir, 'epoch_')
curve_path = os.path.join(dir, 'learning_curve.png')
performance_path = os.path.join(dir, 'performance_')

target_acc = 100.0 * 0.999
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_CLASSES = 4
LEARNING_RATE = 0.00125
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    result_file = open(txt_path, 'w')
    my_print(f"Batch size = {BATCH_SIZE}", result_file)
    result_file.close()

    print("Calculate the number of data...")
    classes, train_nums = cal_num_per_class(train_data_path)
    _, val_nums = cal_num_per_class(val_data_path)
    print('the number of pictures per classes in train dataset:', classes, ':', train_nums)
    print('the number of pictures per classes in validation dataset:', classes, ':', val_nums)

    # net = coatnet_0(NUM_CLASSES)

    # net = SwinTransformer(img_size=224,
    #                       patch_size=8,
    #                       in_chans=3,
    #                       num_classes=NUM_CLASSES,
    #                       embed_dim=128,
    #                       depths=[2, 2, 18, 2],
    #                       num_heads=[4, 8, 16, 32],
    #                       window_size=7,
    #                       mlp_ratio=4.,
    #                       qkv_bias=True,
    #                       qk_scale=None,
    #                       drop_rate=0.0,
    #                       drop_path_rate=0.5,
    #                       ape=False,
    #                       patch_norm=True,
    #                       use_checkpoint=False)

    net = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=NUM_CLASSES)

    #net = Refiner_ViT_S(num_classes=NUM_CLASSES)

    #net = pvig_b_224_gelu()

    #net = deepvit_patch16_224_re_attn_16b()

    #net = CvT(num_classes=NUM_CLASSES)
    
    net = net.to(DEVICE)

    train_iter, test_iter = load_dataset(BATCH_SIZE, train_data_path, val_data_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    record_train, record_test = train(net, train_iter, criterion, optimizer,
                                      NUM_EPOCHS, DEVICE, classes, train_nums, val_nums, lr_scheduler, test_iter, result_file)

    #result_file.close()

    learning_curve(record_train, record_test)

# def Handler(signum, frame):
#     if signum == signal.SIGINT.value:
#         print('SIGINT')
#         result_file.close()
#         sys.exit(1)
# signal.signal(signal.SIGINT, Handler)

def my_print(s, file):
    print(s)
    if file != None:
        print(s, file=file)

def cal_num_per_class(data_path):

    classes = ['ictal', 'normal', 'postictal', 'preictal']
    nums = []

    for class_name in classes: #os.listdir(data_path):
        #classes.append(class_name)
        num=0
        for file in os.listdir( os.path.join(data_path, class_name) ):
            type = filetype.guess( os.path.join(data_path, class_name, file) )
            if type.mime == 'image/png':  # 檔案類型
                num+=1
        nums.append(num)
    
    return classes, nums

def load_dataset(batch_size, train_data_path, val_data_path):
    print("start load_dataset")
    train_set = torchvision.datasets.ImageFolder(
        train_data_path, transform=transforms.Compose(
            [transforms.ToTensor()])

    )
    test_set = torchvision.datasets.ImageFolder(
        val_data_path, transform=transforms.Compose(
            [transforms.ToTensor()])
    )
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    print("finish load_dataset")

    return train_iter, test_iter


def train(net, train_iter, criterion, optimizer, num_epochs, device, classes, train_nums, val_nums,  lr_scheduler=None, test_iter=None, result_file=None):

    net.train()
    record_train = list()
    record_test = list()
    true_label = list()
    pred_label = list()

    num_print = len(train_iter)//4

    for epoch in range(num_epochs):

        result_file = open(txt_path, 'a')  ###############

        my_print("================================ epoch: [{}/{}] ================================".format(epoch + 1, num_epochs), result_file)
        
        total, correct, train_loss = 0, 0, 0
        true_label = list()
        pred_label = list()
        
        start = time.time()

        for i, (X, y) in tqdm(enumerate(train_iter)):
            X, y = X.to(device), y.to(device)
            
            output = net(X)
            #output = output[0]  # Refiner_ViT_train
            _, pred = torch.max(output, 1)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_acc = 100.0 * correct / total

            true_label.extend(y.cpu().numpy())
            pred_label.extend(pred.cpu().numpy())

            if (i + 1) % num_print == 0:
                my_print("step: [{}/{}], train_loss: {:.5f} | train_acc: {:6.4f}% | lr: {:.6f}"
                      .format(i + 1, len(train_iter), train_loss / (i + 1),
                              train_acc, get_cur_lr(optimizer)), result_file)

        record_train.append(train_acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        my_print("----- cost time: {:.4f}s -----".format(time.time() - start), result_file)

        if npy_path is not None:
            np.save(npy_path + str(epoch+1) + '_train_true', true_label)
            np.save(npy_path + str(epoch+1) + '_train_pred', pred_label)

        cal_performance(epoch, 'train', true_label, pred_label, classes, train_nums, result_file)

        #============== test =====================
        if test_iter is not None:
            test_acc = test(net, epoch, test_iter, criterion, device, classes, val_nums, result_file)
            record_test.append(test_acc)

        #============== save weight =====================

        learning_curve(record_train, record_test)

        '''
        best_mAP = train_acc
        best_weight = os.path.join(
            os.path.split(weight_path)[0], "SwinTransformer_224L_Physionet_MergedY_60S_check.pt")
        chkpt = {
            "epoch": epoch,
            "best_mAP": best_mAP,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(chkpt, best_weight)
        
        torch.save(chkpt["model"], best_weight) #看你要不要只存引數
        '''

        torch.save(net.state_dict(), weight_path+"_{}_acc={:.3f}.pt".format(epoch+1, record_test[epoch]))
        print("save weights done")

        result_file.close()


        # =========================== early stop ======================================
        if train_acc >= target_acc and test_acc >= target_acc:
            break
        
        if len(record_test) >= 10 and record_test[-1] < record_test[-5] + 0.5:
            break

        # 預防overfitting
        if len(record_test)>=3 and record_test[-1] < record_test[-2] and record_test[-2] < record_test[-3]:
            break

    torch.save(net.state_dict(), weight_path+"_full.pt")
    
    return record_train, record_test


def test(net, epoch, test_iter, criterion, device, classes, val_nums, result_file=None):
    
    total, correct = 0, 0
    true_label = list()
    pred_label = list()

    net.eval()

    with torch.no_grad():

        my_print("======================== test ========================", result_file)

        for X, y in test_iter:
            X, y = X.to(device), y.to(device)

            output = net(X)
            _, pred = torch.max(output, 1)

            loss = criterion(output, y)

            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()

            true_label.extend(y.cpu().numpy())
            pred_label.extend(pred.cpu().numpy())

    if npy_path is not None:
        np.save(npy_path + str(epoch+1) + '_test_true', true_label)
        np.save(npy_path + str(epoch+1) + '_test_pred', pred_label)

    cal_performance(epoch, 'test', true_label, pred_label, classes, val_nums, result_file)

    test_acc = 100.0 * correct / total


    my_print("=> test_loss: {:.3f} | test_acc: {:6.3f}%"
          .format(loss.item(), test_acc), result_file)

    net.train()

    return test_acc

def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")

    plt.legend(loc=4)
    plt.title("learning curve", fontsize=20)
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(curve_path)

    #plt.show()


def plot_confusion_matrix(epoch, type, confusion_matrix, classes, nums):

    confusion_matrix = confusion_matrix.astype('int')  # / confusion_matrix.sum(axis=1)[:, np.newaxis]  # normalize
    # classes = ["ictal", "normal", "postical", "preictal"]
    df_cm = pd.DataFrame(confusion_matrix, classes, classes)

    s = ''
    for c, n in zip(classes, nums):
        s = s + c + ':' + str(n) + '\n'
    s = s[:-1]  # 去掉最後一個換行

    plt.figure(figsize = (9,6))
    plt.title(type + " : " + "confusion matrix", fontsize=20)
    plt.text(4.75, 0.3, s, size=11, bbox={'boxstyle':'round',
                                            'facecolor':'#FFFFFF',
                                            'edgecolor':'#008B45',
                                            'pad':0.3,
                                            'linewidth':3})
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("True label")
    plt.savefig(cf_path + type + '_' +  str(epoch+1) + '.png')
    plt.close('all')
    #plt.show()

def cal_performance(epoch, type, true_label, pred_label, classes, nums, result_file=None):

    # ================== confusion_matrix =========================
    cf_matrix = confusion_matrix(true_label, pred_label)

    my_print(f'confusion_matrix: \n {cf_matrix}', result_file)
        
    plot_confusion_matrix(epoch, type, cf_matrix, classes, nums)  # plot and save confusion matrix

    # ================== f1-score =========================
    f1_s = f1_score(true_label, pred_label, average='macro')
    #my_print('f1 score : {:.4f}'.format(f1_s), result_file)

    f1_s_per_class = f1_score(true_label, pred_label, average=None)
    #my_print(f'f1 scores_per_class : {f1_s_per_class}', result_file)

    # ================= precision ============================
    PPV = precision_score(true_label, pred_label, average='macro')
    PPV_per_class = precision_score(true_label, pred_label, average=None)

    # ================== sensitivity =========================
    recall = recall_score(true_label, pred_label, average='macro')
    recall_per_class = recall_score(true_label, pred_label, average=None)

    # ================== specificity =========================
    specificity = specificity_score(true_label, pred_label, average='macro')
    specificity_per_class = specificity_score(true_label, pred_label, average=None)

    # ================== FP FN TP TN ===========================
    FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)  
    FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
    TP = np.diag(cf_matrix)
    TN = cf_matrix.sum() - (FP + FN + TP)

    # =================== NPV ======================================
    NPV_per_class = []
    for i in range(len(TN)):
        NPV_per_class.append( TN[i] / (TN[i]+FN[i]) )
    NPV = np.mean(NPV_per_class)

    # ================== csv =======================================
    if performance_path is not None:
        performance = {
            'TP' : TP,
            'TN' : TN,
            'FP' : FP,
            'FN' : FN,
            'Sensitivity(recall)' : recall_per_class,
            'Specificity' : specificity_per_class,
            'Precision(PPV)' : PPV_per_class,
            'NPV' : NPV_per_class,
            'f1 score' : f1_s_per_class
        }

        if classes is not None:
            pfm_df = pd.DataFrame(performance, index=classes)
        else:
            pfm_df = pd.DataFrame(performance)

        avg = pfm_df.mean(0)
        pfm_df.loc['average'] = avg
        
        c = ['Sensitivity(recall)', 'Specificity','Precision(PPV)', 'NPV', 'f1 score']
        for i in range(len(c)):
            pfm_df[c[i]] = pfm_df[c[i]].apply(lambda x: format(x, '.3%'))
        my_print(pfm_df, result_file)

        pfm_df.to_csv(performance_path + type + '_' +  str(epoch+1) + '.csv')

if __name__ == '__main__':

    main()
