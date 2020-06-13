import numpy as np
import argparse
import imutils
import sys
import cv2
from opts import parse_opts
from models.model import generate_model
import torch



def main():
    opts = parse_opts()
    if torch.cuda.is_available():
        opts.cuda = True
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)
    # load the human activity recognition model
    print("[INFO] loading human activity recognition model...")
    # net = cv2.dnn.readNet(args["model"])

    model, parameters = generate_model(opts)
    if opts.resume_path1:
        print('loading checkpoint {}'.format(opts.resume_path1))
        checkpoint = torch.load(opts.resume_path1)
        assert opts.arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    # grab a pointer to the input video stream
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(opts.inputs)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    v_writer = cv2.VideoWriter('mice.avi', fourcc, 10, (1280, 720))

    # loop until we explicitly break from it
    while True:
        # initialize the batch of frames that will be passed through the
        # model
        frames = []
        ori_frames = []
        # loop over the number of required sample frames
        for i in range(0, opts.sample_duration):
            # read a frame from the video stream
            (grabbed, frame) = vs.read()
            
            # if the frame was not grabbed then we've reached the end of
            # the video stream so exit the script
            if not grabbed:
                print("[INFO] no frame read from stream - exiting")
                sys.exit(0)
            ori_frame = frame.copy()
            ori_frames.append(ori_frame)

            # otherwise, the frame was read so resize it and add it to
            # our frames list
            frame = imutils.resize(frame, height=112)
            frames.append(frame)

        # now that our frames array is filled we can construct our blob
        blob = cv2.dnn.blobFromImages(frames, 1.0,
            (opts.sample_size, opts.sample_size), (114.7748, 107.7354, 99.4750),
            swapRB=True, crop=True)
        # blob = cv2.dnn.blobFromImages(frames, 1.0,
        #     (opts.sample_size, opts.sample_size), (0, 0, 0),
        #     swapRB=True, crop=True)

        # img = blob[0, :, :, :]
        # img = np.array(img, dtype=np.uint8)
        # img = np.transpose(img, (1, 2, 0))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # import pdb; pdb.set_trace()
        

        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        # pass the blob through the network to obtain our human activity
        # recognition predictions
        inputs = torch.from_numpy(blob)
        outputs = model(inputs)
        label = np.array(torch.mean(outputs, dim=0, keepdim=True).topk(1)[1].cpu().data[0])
        print(label)
        if label[0] == 0:
            text = 'Normal'
        elif label[0] == 1:
            text = 'Ictal'
        else:
            text = 'NoneType'
        # loop over our frames
        for frame in ori_frames:
            # draw the predicted activity on the frame
            # cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
            v_writer.write(frame)

            # display the frame to our screen
            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    v_writer.release()

if __name__ == "__main__":
    main()