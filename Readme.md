### Implementation of seq2seq using tensorflow=1.4.0 and python=3.6.2
I write a attention-based seq2seq model for neural machine translation. It can runs on multiple GPUs(one PC with multiple GPUs)

#### 1. data
My data was downloaded from nlp.stanford.edu/projects/nmt/. Trained on small dataset english-Vietnamese.

#### 2. script explanation:
0. `build_dict.py`  preprocess dataset. I preprocessed input dataset into pickle files. Transfer string into int32, filter length(3~50). Note your file path.
1. `config.py`      some model parameters.
2. `model_topbah.py`   Bahanau attention on top layer of decoder and encoder
3. `train_vi.py`       entrance for training. set up your own parameters at the beginning of this file. At line 37, set gpu_id like `gpus = "5,6,7"`, No space in string.
4. `gpuloader.py`      dataloader for multiple gpu training.
5. `dataloader.py`     dataloader to feed in data to `tf.placeholder`. Not used.

#### 3. some tips
1. You can try Luong attention as well, but I didn't get well result using Luong, not so easy to train. RMSProp and Adam need small lr(like 0.001), while SGD need bigger like 1.0. But SGD is much harder to train.
2. Best result is `output att=False, rmsp, (lr=0.001, start_decay=8000,0.8)`, got bleu=20.5% on tst2012.vi without beam search.
3. **decode phase not tested.** 
4. Spend lots of time on writing multiple gpu training. how to feed in data? how to compute loss and gradient?


#### 4. to be continue
My code, especially `model.py` and `train.py`, is not well organized, may be updated if I have spare time.

may add comments.

Want to use `tf.data.Dataset` api. I wonder how to set validation_per_train_step.

#### 5. reference
1. https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py       # good for beginners
2. https://github.com/tensorflow/nmt/tree/master/nmt
3. https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
4. Effective Approaches to Attention-based Neural Machine Translation
5. neural machine translation by jointly learning align and translate