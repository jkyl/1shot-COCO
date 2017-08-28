# TextGen
  
TextGen is an image captioning neural network designed for 1-shot learning. This
repository contains code for training and validating the TensorFlow model, as well
as serializing the MS-COCO dataset into TFRecord files

## Contents

### ```textgen.py```
- build and train the image captioning model with optional regularization and
fine-tuning arguments.

### ```prepare_data.sh```
- download, preprocess, and serialize the MS-COCO dataset. Also divides data
into 1-shot and base records.

### ```train.ipynb```
- example notebook for training the model.

### ```validate.ipynb```
- example notebook for generating captions off of which to validate the model.

### ```utils/```
- ### ```models.py```
    - contains ```BaseModel``` class and constituent Keras models
- ### ```preprocess_captions.py```
    - tokenize and save captions to json
- ### ```serialize_data.py```
    - save images and captions to tfrecord files


