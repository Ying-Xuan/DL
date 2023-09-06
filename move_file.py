import os
import shutil
from tqdm import tqdm

src = r'D:\Scalogram\0114_post3\img'
dest = r'D:\Scalogram\img'

classes = os.listdir(src)

for cls in classes:
    print(cls)
    src_dir = os.path.join(src, cls)
    dest_dir = os.path.join(dest, cls)
    for file in tqdm(os.listdir(src_dir)):
        
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
        #print(src_file, dest_file)
        shutil.copyfile(src_file, dest_file)

