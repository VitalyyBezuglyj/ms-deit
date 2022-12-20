# ms-deit
Educational implementation of DeiT using the Mindspore framework

1. `docker build .`
2. `docker run -it -v "path/to/ms-deit":"/home/workdir" image_name`
3. `python preprocess.py` 
4. `python train.py --config src/configs/deit_small_patch16_64.yaml`

Current status
```
Warnings
[WARNING] ME(735:139656560391808,MainProcess):2022-12-20-14:46:24.718.205 [mindspore/train/model.py:1078] For EvaluateCallBack callback, {'epoch_end'} methods may not be supported in later version, Use methods prefixed with 'on_train' or 'on_eval' instead when using customized callbacks.
[WARNING] KERNEL(735,7f0453ac3680,python):2022-12-20-14:48:47.921.019 [mindspore/ccsrc/plugin/device/gpu/kernel/gpu_kernel_factory.cc:151] CheckSM] It is recommended to use devices with a computing capacity >= 7, but the current device's computing capacity is 6. In this case, the computation may not be accelerated. Architectures with TensorCores can be used to speed up half precision operations, such as Volta and Ampere.
[WARNING] MD(735,7f0232ffd700,python):2022-12-20-14:49:03.888.084 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:796] DetectFirstBatch] Bad performance attention, it waits more than 25 seconds and unable to fetch first Batch of data from dataset pipeline, which might result `GetNext` timeout problem. You may test dataset processing performance (with creating dataset iterator) and optimize it. Notes: shuffle operation is turn on for loading Dataset in default, which may effect first batch loading time.
[WARNING] MD(735,7f02337fe700,python):2022-12-20-14:49:04.441.705 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:809] DetectPerBatchTime] Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset pipeline, which might result `GetNext` timeout problem. You may test dataset processing performance(with creating dataset iterator) and optimize it.

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
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 702, in _train_dataset_sink_process
    outputs = train_network(*inputs)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/nn/cell.py", line 596, in __call__
    out = self.compile_and_run(*args)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/nn/cell.py", line 988, in compile_and_run
    return _cell_graph_executor(self, *new_inputs, phase=self.phase)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/common/api.py", line 1192, in __call__
    return self.run(obj, *args, phase=phase)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/common/api.py", line 1229, in run
    return self._exec_pip(obj, *args, phase=phase_real)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/common/api.py", line 98, in wrapper
    results = fn(*arg, **kwargs)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/common/api.py", line 1211, in _exec_pip
    return self._graph_executor(args, phase)
RuntimeError: For 'GetNext', get data timeout. Queue name: 4c8d0410-8075-11ed-ac18-0242ac110006

```

TODO:

0. clean up the code
1. change transforms in `src/data/imagenet.py` -- DONE
2. check whether it works.
3. add `WANDB` logging
4. versioning!