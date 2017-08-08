# download MS-COCO
mkdir data
cd data
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
unzip *.zip
rm *.zip
cd ..

# tokenize
python utils/preprocess_captions.py data/captions_train2014.json data/captions_val2014.json data/preproc/

# serialize training data with half of all classes containing only 2 examples
python utils/serialize_data.py \
    data/train2014 \
    data/preproc/train_captions.json \
    data/instances_train2014.json \
    data/2shot_train.tfrecord \
    --lowshot_value 2
    
# serialize validation data w/o holdouts 
# (still create 3 records though, -lv defaults to inf)
python utils/serialize_data.py \
    data/val2014 \
    data/preproc/val_captions.json \
    data/instances_val2014.json \
    data/2shot_val.tfrecord \
    
