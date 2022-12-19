# ms-deit
Educational implementation of DeiT using the Mindspore framework

1. `docker build .`
2. `docker run -it -v "path/to/ms-deit":"/home/workdir" image_name`
3. `python preprocess.py` 
4. `python train.py --config src/configs/deit_small_patch16_64.yaml`

Current status
```
Traceback (most recent call last):
  File "train.py", line 88, in <module>
    main()
  File "train.py", line 79, in main
    dataset_sink_mode=True)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 1050, in train
    initial_epoch=initial_epoch)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 98, in wrapper
    func(self, *args, **kwargs)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 624, in _train
    cb_params, sink_size, initial_epoch, valid_infos)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 683, in _train_dataset_sink_process
    dataset_helper=dataset_helper)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 441, in _exec_preprocess
    dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 350, in __init__
    self.iter = iterclass(dataset, sink_size, epoch_num)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 568, in __init__
    super().__init__(dataset, sink_size, epoch_num)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 454, in __init__
    dataset)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/_utils.py", line 54, in _get_types_and_shapes
    dataset_types = _convert_type(dataset.output_types())
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/dataset/engine/datasets.py", line 1631, in output_types
    self.saved_output_types = runtime_getter[0].GetOutputTypes()
RuntimeError: Unexpected error. map operation: [TypeCast] failed. The corresponding data files: /home/workdir/src/data/d_train.csv. TypeCast: TypeCast does not support cast from string to int32
Line of code : 369
File         : mindspore/ccsrc/minddata/dataset/kernels/data/data_utils.cc

```

TODO:

1. change transforms in `src/data/imagenet.py`
2. check whether it works.
3. add `WANDB` logging
4. versioning!