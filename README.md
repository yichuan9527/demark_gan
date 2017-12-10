# De-mark GAN

---

`data_read.py` is the data reading file.  
`gen_image.py` uses the trained de-mark GAN model to generate clean ID photos.  
`module.py` is the demark GAN model file.  
`pre_model.py` is the pre-trained resnet model file. You can use the file trained your own resnet model on the [mscelba dataset](http://www.msceleb.org)  to compute the feature loss.
'main.py' is the train file.  
first setp: use the `pre_model.py` train the resnet on the mscelba dataset
second step: change the path in 'main.py' and  'data_read.py' to train the De-mark GAN model in you own datasets.

--
##Prerequisites

 - python 2.7
 - pytorch 2.0
 - PIL
 - pickle

