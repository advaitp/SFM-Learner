# SFMLearner 
## Overview 
There has been a lot of work in the field of structure from motion using deep learning methods one such paper Unsupervised Learning of Depth and Ego-Motion from Video deals with the same.In this project we explore the paper and we try to improve the network using various techniques.Few methods that we implemented worked and gave improved results while few others did not work.In this project we explain each techniques and it’s effect on the network and the output.

# ![1](https://github.com/advaitp/SFM-Learner/blob/main/Images/sfmlearnerarch.png)

### To run the code
Install manually the following packages :

```
pytorch >= 1.0.1
pebble
matplotlib
imageio
scipy
scikit-image
argparse
tensorboardX
blessings
progressbar2
path.py
```
### Note
Because it uses latests pytorch features, it is not compatible with anterior versions of pytorch.

If you don't have an up to date pytorch, the tags can help you checkout the right commits corresponding to your pytorch version.

### What has been done

* Training has been tested on KITTI and CityScapes.
* Dataset preparation has been largely improved, and now stores image sequences in folders, making sure that movement is each time big enough between each frame
* That way, training is now significantly faster, running at ~0.14sec per step vs ~0.2s per steps initially (on a single GTX980Ti)
* In addition you don't need to prepare data for a particular sequence length anymore as stacking is made on the fly.
* You can still choose the former stacked frames dataset format.
* Convergence is now almost as good as original paper with same hyper parameters
* You can know compare with ground truth for your validation set. It is still possible to validate without, but you now can see that minimizing photometric error is not equivalent to optimizing depth map.


## To train the model 

```
python Train.py 

parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='stacked', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) '
                    'sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use depth ground truth for validation. '
                    'You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--with-pose', action='store_true', help='use pose ground truth for validation. '
                    'You need to store it in text files of 12 columns see data/kitti_raw_loader.py for an example '
                    'Note that for kitti, it is recommend to use odometry train set to test pose')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-f', '--training-output-freq', type=int,
                    help='frequence for outputting dispnet outputs and warped imgs at training for all scales. '
                         'if 0, will not output',
                    metavar='N', default=0)

```

## To test the model 
```
python Test.py
parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox'])
parser.add_argument("--gps", '-g', action='store_true',
                    help='if selected, will get displacement from GPS for KITTI. Otherwise, will integrate speed')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

## Results 
# ![2](https://github.com/advaitp/SFM-Learner/blob/main/Images/sfmlearnermodels.png)
