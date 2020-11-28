import os
import random 


if __name__ == "__main__":
    video_list_path = 'dataset/mice_labels/YJ_mice_015_0525.txt'
    origin_v_path = ''
    target_v_path = ''
    with open(video_list_path, 'r') as file:
        video_list = list(file)
    v_num = len(video_list)
    random.shuffle(video_list)
    ratio = 0.25
    new_v_list = video_list[:int(v_num*ratio)]
    test_v_list = video_list[int(v_num*ratio):]
    # old_v_path = '/home/alien/zhangyujia/datasets/mice/YJ_015_2020-05-25'
    # new_v_path = '/home/alien/zhangyujia/datasets/mice/mice_first'
    log_path_train = 'dataset/mice_labels/train_YJ_015_0525.txt'
    log_path_test = 'dataset/mice_labels/test_YJ_015_0525.txt'
    with open(log_path_train, 'w') as file:
        for line in new_v_list:
            file.write(line)
    with open(log_path_test, 'w') as file:
        for line in test_v_list:
            file.write(line)

    # for line in new_v_list:
    #     video_name = line.split(' ')[0].split('/')[1]
    #     os.system('cp {} {}'.format(
    #         os.path.join(old_v_path, 'pre_control', video_name+'.mp4'),
    #         os.path.join(new_v_path, 'pre_control', video_name+'.mp4')))
    

    

