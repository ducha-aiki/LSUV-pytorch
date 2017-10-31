# Layer-sequential unit-variance (LSUV) initialization for PyTorch

This is sample code for LSUV and initializations, implemented in python script within PyTorch framework. 

Usage:

    from LSUV import LSUVinit
    ...
    model = LSUVinit(model,data)

See detailed example in [example.py](example.py)

LSUV initialization is described in:

Mishkin, D. and Matas, J.,(2015). All you need is a good init. ICLR 2016 [arXiv:1511.06422](http://arxiv.org/abs/1511.06422).

Original Caffe implementation  [https://github.com/ducha-aiki/LSUVinit](https://github.com/ducha-aiki/LSUVinit)

Torch re-implementation [https://github.com/yobibyte/torch-lsuv](https://github.com/yobibyte/torch-lsuv)

Keras implementation: [https://github.com/ducha-aiki/LSUV-keras](https://github.com/ducha-aiki/LSUV-keras)

**New!** Thinc re-implementation [LSUV-thinc](https://github.com/explosion/thinc/blob/e653dd3dfe91f8572e2001c8943dbd9b9401768b/thinc/neural/_lsuv.py)
