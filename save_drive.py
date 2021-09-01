import argparse
import time
#import json
#import collections
#import os
import shutil
#import pickle as pkl
from glob import glob

def main(args):

    for i in range(20):
        pf_list = glob(args.in_path+"model.bi*")
        f_list = [f.split('/')[-1] for f in pf_list]
        for j in range(2):
            shutil.copy(pf_list[j],args.out_path)
        print(f"saved : {time.time()}")
        time.sleep(1800)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='program to preproceed')
    parser.add_argument('--in_path', default='', 
                        help='root of the data')
    parser.add_argument('--out_path', default="/content/drive/MyDrive/Cur_NMT/bin_0816/",
                        help='destination')
    parser.add_argument('--dscd', action='store_true',
                        help='for descending ordering')

    args = parser.parse_args()
    
    main(args)

# python3 save_in_drive.py --in_path=../pathto/extractor/model/ckpt/ --out_path=../../drive/MyDrive/fast_abs_folder/extr/ko_model/ckpt
# python3 save_in_drive.py --in_path=../pathto/abstractor/model/ckpt/ --out_path=../../drive/MyDrive/fast_abs_folder/abst/ko_model2/ckpt
# python3 save_in_drive.py --in_path=../pathto/save/model/ckpt/ --out_path=../../drive/MyDrive/fast_abs_folder/full/ko_model/ckpt --dscd

