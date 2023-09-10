#/bin/bash

mkdir dataset
cd dataset

# download from roboflow. The key depends on your account.
# https://public.roboflow.com/object-detection/pascal-voc-2012/1
# curl -L "https://public.roboflow.com/ds/cqVKG2FrTp?key=hogehoge" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
echo "check the number of images"
echo "train"
find train | wc
echo "valid"
find valid | wc

cd ../

python3 utils/one_person_dataset.py

# coco test data
# wget -nc http://images.cocodataset.org/zips/test2014.zip
# wget -nc http://images.cocodataset.org/annotations/image_info_test2014.zip
# unzip -n test2014.zip
# unzip -n image_info_test2014.zip
