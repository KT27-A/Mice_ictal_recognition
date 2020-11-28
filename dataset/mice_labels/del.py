import os


def del_long():
    new_list = []
    with open('dataset/mice_labels/train_dataset_5.txt', 'r') as f:
        list_input = list(f)

    for i, line in enumerate(list_input):
        try:
            name_i, class_i = line.strip('\n').split(' ')
        except:
            import pdb; pdb.set_trace()
        if class_i == '#0':
            if i < len(list_input)-1:
                if name_i+'_1' != list_input[i+1].strip('\n').split(' ')[0]:
                    new_list.append(line)
                else:
                    # import pdb; pdb.set_trace()
                    pass
            # if 'pre_control/2020-03-' in name_i:
            #     if len(name_i.split('/')[1]) > 16:
            #         new_list.append(i)
            # elif 'P_016_YJ_LJ_2020-06-' in name_i:
            #     if len(name_i.split('/')[1]) > 28:
            #         new_list.append(i)
            # elif 'P_024_FP_YJ_MF_2020-06-' in name_i:
            #     if len(name_i.split('/')[1]) > 31:
            #         new_list.append(i)
            # elif 'P_025_FP_YJ_MF_2020-06' in name_i:
            #     if len(name_i.split('/')[1]) > 31:
            #         new_list.append(i)
        else:
            new_list.append(line)
    
    with open('dataset/mice_labels/train_dataset_5_new.txt', 'w') as f:
        for i in new_list:
            f.write(i)

def del_no_frame():
    new_list = []
    with open('dataset/mice_labels/train_dataset_6.txt', 'r') as f:
        list_input = list(f)
    path = '../datasets/mice/dataset_5_1f_256'
    for fold in list_input:
        item = os.listdir(os.path.join(path, fold.split(' ')[0]))
        if item:
            new_list.append(fold)
        else:
            print('empty')

    with open('dataset/mice_labels/train_dataset_6_new.txt', 'w') as f:
        for i in new_list:
            f.write(i)





if __name__ == "__main__":
    del_no_frame()

            