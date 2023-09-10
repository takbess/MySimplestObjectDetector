#/bin/bash

cd dataset

# download from roboflow. The key depends on your account.
# https://public.roboflow.com/object-detection/pascal-voc-2012/1
curl -L "https://public.roboflow.com/ds/cqVKG2FrTp?key=hogehoge" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
# tar xvf dataset/VOCdevkit_18-May-2011.tar
# tar xvf VOCtrainval_11-May-2012.tar

# coco test data
# wget -nc http://images.cocodataset.org/zips/test2014.zip
# wget -nc http://images.cocodataset.org/annotations/image_info_test2014.zip
# unzip -n test2014.zip
# unzip -n image_info_test2014.zip
