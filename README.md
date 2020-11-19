Based on https://github.com/DLR-RM/AugmentedAutoencoder <br/>
It only includes the Augmented Autoencoder part and does not include object detection part.<br/>
For SSD object detection, please check `docker_ssd` repository.<br/>
The docker image meets the software requirements.
# 







## Inference
Option 2: Using the Google Detection API with Fixes
Train a 2D detector following https://github.com/naisy/train_ssd_mobilenet
adapt /auto_pose/test/googledet_utils/googledet_config.yml
When you are running `python auto_pose/test/aae_googledet_webcam_multi.py exp_group/my_autoencoder`, the error of `error: (-215:Assertion failed) (mtype == CV_8U || mtype == CV_8S) && _mask.sameSize(*psrc1) in function 'binary_op'` shows up. That means the video resolution in `/auto_pose/test/googledet_utils/googledet_config.ym` does not match the video resolution.

