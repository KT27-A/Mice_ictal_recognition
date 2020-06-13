import glob
import os
import random



def make_train_test():
    train_name = 'train_mice.txt'
    test_name = 'test_mice.txt'
    epilepsy_path = 'pre_case'
    normal_path = 'pre_control'
    train_ratio = 0.75
    
    train_epi_samples = []
    test_epi_samples = []
    for ori_path, fold, items in os.walk(epilepsy_path):
        video_num = len(items)
        random.shuffle(items)
        train_epi_samples = sorted(items[:int(video_num*train_ratio)])
        test_epi_samples = sorted(items[int(video_num*train_ratio):])

    for ori_path, fold, items in os.walk(normal_path):
        video_num = len(items)
        random.shuffle(items)
        train_norm_samples = sorted(items[:int(video_num*train_ratio)])
        test_norm_samples = sorted(items[int(video_num*train_ratio):])

    print('Making train list')
    file = open(train_name, 'w')
    for i in train_norm_samples:
        line = os.path.join(normal_path, i[:-4])+' #0'
        file.write(line)
        file.write('\n')
    for i in train_epi_samples:
        line = os.path.join(epilepsy_path, i[:-4])+' #1'
        file.write(line)
        file.write('\n')
    print('Train list done')

    print('Making test list')
    file = open(test_name, 'w')
    for i in test_norm_samples:
        line = os.path.join(normal_path, i[:-4])+' #0'
        file.write(line)
        file.write('\n')
    for i in test_epi_samples:
        line = os.path.join(epilepsy_path, i[:-4])+' #1'
        file.write(line)
        file.write('\n')
    print('Test list done')


def make_new_test():
    test_name = 'test_mice_0320.txt'
    dir_path = 'mice_2020-03-20'
    case_path = 'pre_case'
    control_path = 'pre_control'

    v_case_spl = sorted(os.listdir(os.path.join(dir_path, case_path)))
    v_control_spl = sorted(os.listdir(os.path.join(dir_path, control_path)))

    print('Making list {}'.format(test_name))
    file = open(test_name, 'w')
    for i in v_control_spl:
        line = os.path.join(control_path, i[:-4]) + ' #0'
        file.write(line)
        file.write('\n')

    for i in v_case_spl:
        line = os.path.join(case_path, i[:-4])+' #1'
        file.write(line)
        file.write('\n')
    file.close()
    print('list {} is done'.format(test_name))




def main():
    # make_train_test()
    make_new_test()
    

if __name__ == "__main__":
    main()