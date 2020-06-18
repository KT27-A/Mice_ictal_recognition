# Mice_ictal_recognition
Recognizing mice ictal

For real-time application
1. Download models from URL
https://drive.google.com/file/d/11LcXG6e9mM_TbeN8iY1CJVlokP8xX9lo/view?usp=sharing, https://drive.google.com/file/d/1Cag9Zr0HvJzWRy839qgaBAMmj1b-TdcX/view?usp=sharing, https://drive.google.com/file/d/1WqqDb6i-lLCwGoYFfdoUthNUaSrugfh7/view?usp=sharing

2. Place the models in the right path \
RGB_Kinetics_64f -> ../pretrained_model \
model_mice.pth -> ./results_mice_resnext101

Using command in comment.txt 
```
-------------------------------real_time_resnext101_mice------------------------------------
python real_time_demo.py --n_classes 2 --model resnext --model_depth 101 \
--sample_duration 64 --annotation_path "dataset/mice_labels" \
--resume_path1 "results_mice_resnext101/save_71_max.pth" \
--inputs "07_51-52.mp4" 
```

For training
1. Using make_list.py to make the list of training and testing;
2. Placing generated training/testing list to dataset/mice_labels
3. Using extract_frames.py to transform videos to frames
4. Using command in comment.txt for training

For testing online (.mp4 file)

Using command in comment.txt

python test_plus_mice_online.py --batch_size 1 --n_classes 2 --model resnext --model_depth 101 \
--log 1 --dataset MICE_online --modality RGB --sample_duration 64 --split 1 --only_RGB  \
--resume_path1 "results_mice_resnext101/model_mice.pth" \
--frame_dir "/home/katou2/github_home/0609_epilepsy/mice" \
--annotation_path "dataset/mice_labels" \
--result_path "results_mice_resnext101/" \
--n_workers 0 --test_file 'test_mice.txt'


![grab-landing-page](https://github.com/Katou2/Mice_ictal_recognition/blob/master/demo.gif)
