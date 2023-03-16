# DFlow_BasicSR
Official code for "Propagating difference flows for efficient video super-resolution" [BMVC22] 
[**Paper**](https://bmvc2022.mpi-inf.mpg.de/0060.pdf)

The code is based on the the BasicSR framework. [**Link**](https://github.com/XPixelGroup/BasicSR)

## Training
```
cd DFlow_BasicSR
python setup.py develop
python basicsr/train.py -opt /path/to/DFlow_BasicSR/options/train/train_code_Vimeo90K_BIx4.yml  (stage1)
python basicsr/train.py -opt /path/to/DFlow_BasicSR/options/train/trains2_code_Vimeo90K_BIx4.yml (stage2)
```

## Testing
```
cd DFlow_BasicSR
python setup.py develop
python basicsr/train.py -opt /path/to/DFlow_BasicSR/options/test/test_code_DFlow_BIx4.yml
```

## Acknowledgement
Thanks to the BasicSR framework for convenient experiment tracking.
