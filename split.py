import os
import shutil

def split_bacteria_and_virus(root,brige= '_'):
    imgs = os.listdir(root)
    bac_list = []
    virus_list = []
    for img in imgs:
        name = img.split(brige)
        if 'bacteria' in name:
            bac_list.append(img)
        elif 'virus' in name:
            virus_list.append(img)
    
    bac_pathes = [os.path.join(root,i) for i in bac_list]
    virus_pathes = [os.path.join(root,i) for i in virus_list]
    dir_name_bac = os.path.join(os.path.dirname(root),'Bacteria_PNEUMONIA')
    dir_name_virus = os.path.join(os.path.dirname(root),'Virus_PNEUMONIA')
    # move bacteria
    if os.path.isdir(dir_name_bac):
        for path in bac_pathes:
            shutil.move(path,dir_name_bac)
    else:
        os.mkdir(dir_name_bac)
        for path in bac_pathes:
            shutil.move(path,dir_name_bac)
    # move virus
    if os.path.isdir(dir_name_virus):
        for path in virus_pathes:
            shutil.move(path,dir_name_virus)
    else:
        os.mkdir(dir_name_virus)
        for path in virus_pathes:
            shutil.move(path,dir_name_virus)

    print('==finished==')

if  __name__ == '__main__':
    path = r'E:\Ying Xuan\class\數據科學方法\computer assignment2\chest_xray\test\PNEUMONIA'
    split_bacteria_and_virus(root=path)