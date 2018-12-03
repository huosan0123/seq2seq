### Implementation of seq2seq using tensorflow=1.4 and python3
I am training a seq2seq-based model for my own task.

The problem is :
When running "python train.py", at first the training speed is about 1.5min/20 iteration (you can see this in 181203.log). Then, speed decreased to 4min / 20 iteration. 

Files that I used are: 
- train.py
- dataloder.py
- model.py
- config.py

My model.py(it not complete) now only care about training phase, so forget about the decode phase.
