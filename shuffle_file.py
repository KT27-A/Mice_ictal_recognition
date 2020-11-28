import random

if __name__ == "__main__":
    with open('dataset/mice_labels/test_control_026_0626.txt', 'r') as f:
        data = f.readlines()
    for i in range(10):
        random.shuffle(data)
    with open('dataset/mice_labels/test_control_026_0626_s_0.5.txt', 'w') as f:
        for line in data:
            f.write(line)
    