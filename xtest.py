import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import os
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
#from Swin_Transformer.Swin_Transformer_main.models.swin_transformer import SwinTransformer
#from ConvMixer.convmixer_main.convmixer import *
#from ConvNeXt.ConvNeXt_main.models.convnext import *

# 驗證資料路徑
val_data_path = r"D:\Scalogram\張珮禎\img"

weight_path = r'D:\Scalogram\詹焜然\weigth\CoAtnet\epoch__full.pt'

dir = r'D:\Scalogram\張珮禎\weight\be_test\CoAtNet'  # 實驗數據資料夾
txt_path = os.path.join(dir, 'predic_result.txt')
cf_path = os.path.join(dir, 'confusion_matrix_')  # 混淆矩陣路徑
npy_path = os.path.join(dir, 'test_')  # true label、pred label 路徑
performance_path = os.path.join(dir, 'performance.csv')

BATCH_SIZE = 1
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    print('Calculate the number...')
    classes, val_nums = cal_num_per_class(val_data_path)
    print('the number of pictures per classes in validation dataset:', classes, ':', val_nums)

    # net = convnext_xlarge()
    
    #net = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=NUM_CLASSES)

    net = coatnet_0(NUM_CLASSES)

    # net = SwinTransformer(img_size=224,
    #                       patch_size=4,
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

    net = net.to(DEVICE)

    test_iter = load_dataset(BATCH_SIZE, val_data_path)

    net.load_state_dict( torch.load(weight_path, map_location=DEVICE) )

    result_file = open(txt_path, 'w')

    test(net, test_iter, DEVICE, classes, val_nums, result_file)

    result_file.close()


def my_print(s, file):
    print(s)
    if file != None:
        print(s, file=file)

def cal_num_per_class(data_path):

    classes = ['ictal', 'normal', 'postictal', 'preictal']
    nums = []

    for class_name in classes: #os.listdir(data_path).sort():
        #classes.append(class_name)
        num=0
        for file in os.listdir( os.path.join(data_path, class_name) ):
            type = filetype.guess( os.path.join(data_path, class_name, file) )
            if type.mime == 'image/png':  # 檔案類型
                num+=1
        nums.append(num)
    
    return classes, nums

def load_dataset(batch_size, val_data_path):
    
    print("start load_dataset")

    test_set = torchvision.datasets.ImageFolder(
        val_data_path, transform=transforms.Compose(
            [transforms.ToTensor()])
    )

    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    print("finish load_dataset")

    return test_iter

def test(net, test_iter, device, classes=None, val_nums=None, result_file=None):
    
    total, correct = 0, 0
    true_label = list()
    pred_label = list()

    net.eval()

    with torch.no_grad():

        my_print("======================== test ========================", result_file)

        for X, y in tqdm(test_iter):

            X, y = X.to(device), y.to(device)

            output = net(X)
            _, pred = torch.max(output, 1)

            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()

            true_label.extend(y.cpu().numpy())
            pred_label.extend(pred.cpu().numpy())

    if npy_path is not None:
        np.save(npy_path + '_true', true_label)
        np.save(npy_path + '_pred', pred_label)
                
    cal_performance('train', true_label, pred_label, classes, val_nums, result_file)

    test_acc = 100.0 * correct / total
    my_print(test_acc, result_file)


    net.train()

    return test_acc

def plot_confusion_matrix(type, confusion_matrix, classes=None, nums=None):

    #confusion_matrix = confusion_matrix.astype('int')  # / confusion_matrix.sum(axis=1)[:, np.newaxis]  # normalize
    
    #confusion_matrix = confusion_matrix[0:4, 0:4]  #####

    df_cm = pd.DataFrame(confusion_matrix, classes, classes)

    s = ''
    if classes is not None and nums is not None:
        for c, n in zip(classes, nums):
            s = s + c + ':' + str(n) + '\n'
        s = s[:-1]  # 去掉最後一個換行

    plt.figure(figsize = (9,6))
    plt.title(type + " : " + "confusion matrix", fontsize=20)
    plt.text(4.75, 0.3, s, size=10, bbox={'boxstyle':'round',
                                            'facecolor':'#FFFFFF',
                                            'edgecolor':'#008B45',
                                            'pad':0.3,
                                            'linewidth':3})
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("True label")
    plt.savefig(cf_path + type + '.png')
    plt.close('all')
    #plt.show()

def cal_performance(type, true_label, pred_label, classes=None, nums=None, result_file=None):

    # ================== confusion_matrix =========================
    cf_matrix = confusion_matrix(true_label, pred_label)
    my_print(f'confusion_matrix: \n {cf_matrix}', result_file)

    if cf_path is not None:
        plot_confusion_matrix(type, cf_matrix, classes, nums)  # plot and save confusion matrix

    # ================== f1-score =========================
    f1_s = f1_score(true_label, pred_label, average='macro')
    my_print('f1 score : {:.4f}'.format(f1_s), result_file)

    f1_s_per_class = f1_score(true_label, pred_label, average=None)
    my_print(f'f1 scores_per_class : {f1_s_per_class}', result_file)

    # ================= precision ============================
    PPV = precision_score(true_label, pred_label, average='macro')
    my_print('PPV : {:.4f}'.format(PPV), result_file)

    PPV_per_class = precision_score(true_label, pred_label, average=None)
    my_print(f'PPV_per_class : {PPV_per_class}', result_file)

    # ================== sensitivity =========================
    recall = recall_score(true_label, pred_label, average='macro')
    my_print('sensitivity : {:.4f}'.format(recall), result_file)

    recall_per_class = recall_score(true_label, pred_label, average=None)
    my_print(f'sensitivity_per_class : {recall_per_class}', result_file)

    # ================== specificity =========================
    specificity = specificity_score(true_label, pred_label, average='macro')
    my_print('specificity : {:.4f}'.format(specificity), result_file)

    specificity_per_class = specificity_score(true_label, pred_label, average=None)
    my_print(f'specificity_per_class : {specificity_per_class}', result_file)

    # ================== FP FN TP TN ===========================
    FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)  
    FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
    TP = np.diag(cf_matrix)
    TN = cf_matrix.sum() - (FP + FN + TP)
    my_print(f'FP : {FP}  FN:{FN}  TP:{TP}  TN:{TN}', result_file)

    # =================== NPV ======================================
    NPV_per_class = []
    for i in range(len(TN)):
        NPV_per_class.append( TN[i] / (TN[i]+FN[i]) )
    NPV = np.mean(NPV_per_class)
    my_print('NPV : {:.4f}'.format(NPV), result_file)
    my_print(f'NPV_per_class : {NPV_per_class}', result_file)

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
        print(pfm_df)
        pfm_df.to_csv(performance_path)

if __name__ == '__main__':

    main()
