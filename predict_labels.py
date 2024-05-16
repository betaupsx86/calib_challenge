import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from argparse import ArgumentParser

from camera_geometry import CameraGeometry
from calibrated_lane_detector import CalibratedLaneDetector

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_config', help='Model config file')
    parser.add_argument('--model_checkpoint', help='Model checkpoint file')
    parser.add_argument('--model_device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--vid', help='Video file')
    parser.add_argument('--vid_start_frame', type=int, default=None, help='Video start frame')
    parser.add_argument('--vid_stop_frame', type=int, default=None, help='Video stop frame')
    parser.add_argument('--out-dir', default='.', help='Path to output folder')
    args = parser.parse_args()
    return args
      
def main(args):
    filename = os.path.splitext(os.path.basename(args.vid))[0]
    filename_labels = "{}/{}.txt".format(os.path.dirname(args.vid), filename)
    try:
        gt_labels = np.loadtxt(filename_labels)
    except Exception:
        gt_labels = None
    
    # Store video properties
    cap = cv2.VideoCapture(args.vid)
    video_properties = dict()
    video_properties["width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_properties["height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_properties["fps"] = cap.get(cv2.CAP_PROP_FPS)

    try:
        os.makedirs(args.out_dir)
    except FileExistsError:
        None
    vid_out_file = "{}/{}.avi".format(args.out_dir, filename)
    vid_out_labels = "{}/{}.txt".format(args.out_dir, filename)
    output = cv2.VideoWriter(vid_out_file, cv2.VideoWriter_fourcc('M','J','P','G'), video_properties["fps"], (int(video_properties["width"]), int(video_properties["height"])))

    focal_length_pix = 910
    camera_geometry = CameraGeometry(image_width_pix=int(video_properties["width"]), image_height_pix=int(video_properties["height"]), focal_length_pix=focal_length_pix)
    calibrated_detector = CalibratedLaneDetector(args.model_config, args.model_checkpoint, args.model_device, cam_geom = camera_geometry)

    yaws, pitches = [],[]

    frame_ct=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or (args.vid_stop_frame and frame_ct > args.vid_stop_frame):
            break
        if args.vid_start_frame and frame_ct < args.vid_start_frame:
            frame_ct+=1
            continue
        yaws.append(np.nan)
        pitches.append(np.nan)

        fits, preds = calibrated_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = calibrated_detector.draw_lanes_and_vanishing_point(frame, preds)

        p, y = calibrated_detector.get_estimated_pitch_yaw_rad()
        pitches[-1] = -p
        yaws[-1] = -y

        if gt_labels is not None and frame_ct < len(gt_labels):
            print("Processed frame {}: pitch {} yaw {} pitch_gt {} yaw_gt {}".format(frame_ct, pitches[-1], yaws[-1], gt_labels[frame_ct][0], gt_labels[frame_ct][1]))
        else:
            print("Processed frame {}: pitch {} yaw {}".format(frame_ct, pitches[-1], yaws[-1]))

        frame_ct+=1

        if output:
            output.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # The following frees up resources and closes all windows
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    try:
        np.savetxt(vid_out_labels, np.array([pitches, yaws]).T)
    except Exception:
        print("Could not save labels")

    if gt_labels is not None and len(gt_labels):
        fig, ax = plt.subplots(2)
        ax[0].plot([i for i in range(len(pitches))], pitches, label='predicted')
        ax[0].plot([i for i in range(len(gt_labels))], gt_labels[:,0], label='ground truth')
        ax[0].legend(shadow=True, fancybox=True)

        ax[1].plot([i for i in range(len(yaws))], yaws, label='predicted')
        ax[1].plot([i for i in range(len(gt_labels))], gt_labels[:,0], label='ground truth')
        ax[1].legend(shadow=True, fancybox=True)

        ax[0].set(ylabel='pitch', xlabel='frame')
        ax[0].label_outer()
        ax[1].set(ylabel='yaw', xlabel='frame')
        ax[1].label_outer()

        fig.canvas.draw()
        
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        plot_out_file = "{}/{}_plot.jpg".format(args.out_dir, filename)
        cv2.imwrite(plot_out_file, image)

if __name__ == '__main__':
    args = parse_args()
    main(args)
