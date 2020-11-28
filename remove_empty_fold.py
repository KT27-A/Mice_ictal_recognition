import os

if __name__ == "__main__":
    list_path = 'dataset/mice_labels/train_dataset_7.txt'
    up_dir_path = '/home/alien/zhangyujia/datasets/mice/dataset_7_1f_256'
    new_list = []
    with open(list_path, 'r') as f:
        dir_list = list(f)
    for dir_path_i in dir_list:
        dir_path = os.path.join(up_dir_path, dir_path_i.strip('\n').split(' #')[0])
        item = os.listdir(dir_path)
        if item:
            new_list.append(dir_path_i)
        else:
            print(dir_path_i.strip('\n').split(' #')[0])
    with open('dataset/mice_labels/train_dataset_7_new.txt', 'w') as f:
        for d in new_list:
            f.write(d)
        
        