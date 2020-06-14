# Mice_ictal_recognition
Recognizing mice ictal

For real-time application
1. Download models from URL
https://portland-my.sharepoint.com/personal/yzhang2383-c_ad_cityu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyzhang2383%2Dc%5Fad%5Fcityu%5Fedu%5Fhk%2FDocuments%2Ftrained%5Fmodel&originalPath=aHR0cHM6Ly9wb3J0bGFuZC1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC95emhhbmcyMzgzLWNfYWRfY2l0eXVfZWR1X2hrL0VsTXk2eUtpTWFwRXZFamRNWUFjVkNZQkc0ZW1GMDFxR2NnT2w5UDh2Y3hITVE_cnRpbWU9cEVVaGlhRVAyRWc

2. Place the models in the right place
RGB_Kinetics_64f -> ../pretrained_model
save_71_max.pth -> ./results_mice_resnext101

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

![grab-landing-page](https://github.com/Katou2/Mice_ictal_recognition/blob/master/demo.gif)
