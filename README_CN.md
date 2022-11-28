# 目录

<!-- TOC -->

- [目录](#目录)
- [deit描述](#deit描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的deit训练](#ImageNet-1k上的deit训练)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的deit推理](#ImageNet-1k上的deit推理)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [deit描述](#目录)

RepVGG是由清华大学&旷世科技等提出的一种新颖的CNN设计范式，它将ACNet的思想与VGG架构进行了巧妙的结合，即避免了VGG类方法训练所得精度低的问题，又保持了VGG方案的高效推理优点，它首次将plain模型的精度在ImageNet上提升到了超过80%top1精度。相比ResNet、RegNet、EfficientNet等网络，RepVGG具有更好的精度-速度均衡。

# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/enable_mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```text
├── convnext
  ├── README_CN.md                        // RepVGG相关说明
  ├── scripts
      ├──run_standalone_train_ascend.sh   // 单卡Ascend910训练脚本
      ├──run_distribute_train_ascend.sh   // 多卡Ascend910训练脚本
      ├──run_infer_310.sh                 // Ascend310推理脚本
      ├──run_eval_ascend.sh               // 测试脚本
  ├── src
      ├──configs                          // Deit的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // 模型定义文件夹
          ├──deit.py                      // deit模型文件
          ├──initializer.py               // 初始化函数
          ├──vision_transformer.py        // 定义ViT文件
          ┕──layers                       // 相关工具层定义文件
  │   ├──trainers                         // 自定义TrainOneStep文件
  │   ├──tools                            // 工具文件夹
          ├──callback.py                  // 自定义回调函数，训练结束测试
          ├──cell.py                      // 一些关于cell的通用工具函数
          ├──criterion.py                 // 关于损失函数的工具函数
          ├──get_misc.py                  // 一些其他的工具函数
          ├──optimizer.py                 // 关于优化器和参数的函数
          ┕──schedulers.py                // 学习率衰减的工具函数
  ├── train.py                            // 训练文件
  ├── eval.py                             // 评估文件
  ├── export.py                           // 模型导出文件
  ├── preprocess.py                       // 推理数据集与处理文件
  ├── postprocess.py                      // 推理精度处理文件
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置RepVGG和ImageNet-1k数据集。

  ```text
    # Architecture 79.9%
    arch: deit_small_patch16_224        # deit结构选择
    # ===== Dataset ===== #
    data_url: ./data/imagenet           # 数据集地址
    set: ImageNet                       # 数据集名字
    num_classes: 1000                   # 数据集分类数目
    mix_up: 0.8                         # MixUp数据增强参数
    cutmix: 1.0                         # CutMix数据增强参数
    auto_augment: rand-m9-mstd0.5-inc1  # AutoAugment参数
    interpolation: bicubic              # 图像缩放插值方法
    re_prob: 0.25                       # RandomErasing概率
    re_mode: pixel                      # RandomErasing模式
    re_count: 1                         # RandomErasing重复次数
    mixup_prob: 1.                      # MixUp概率
    switch_prob: 0.5                    # MixUp和CutMix切换概率
    mixup_mode: batch                   # MixUp模式
    image_size: 224                     # 训练图像大小
    crop_pct: 0.9                       # 图像缩放比例
    # ===== Learning Rate Policy ======== #
    optimizer: adamw                    # 优化器类别
    base_lr: 0.0005                     # 基础学习率
    warmup_lr: 0.000001                 # 学习率热身初始化学习率
    min_lr: 0.000001                    # 最小学习率
    lr_scheduler: cosine_lr             # 学习率变化趋势
    warmup_length: 5                    # 学习率热身轮数
    # ===== Network training config ===== #
    amp_level: O1                       # 混合精度等级
    beta: [ 0.9, 0.999 ]                # 优化器参数
    clip_global_norm_value: 5.          # 全局梯度截断
    is_dynamic_loss_scale: True         # 是否使用动态缩放
    epochs: 300                         # 训练轮数
    cooldown_epochs: 10                 # 学习率冷却轮数
    label_smoothing: 0.1                # 标签平滑系数
    weight_decay: 0.05                  # 权重衰减
    momentum: 0.9                       # 动量参数
    batch_size: 128                     # 单卡批次
    drop_path_rate: 0.1                 # 丢弃参数
    # ===== EMA ===== #
    with_ema: False                     # 是否使用ema更新
    ema_decay: 0.9999                   # ema移动系数
    # ===== Hardware setup ===== #
    num_parallel_workers: 16            # 数据预处理线程数
    device_target: Ascend               # GPU或者Ascend
  ```

更多配置细节请参考脚本`config.py`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/deit_small_patch16_224.yaml \
  > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/deit_small_patch16_224.yaml \
  --pretrained ./ckpt_0/deit_small_patch16_224.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

  [hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 导出过程

### 导出

  ```bash
  python export.py --pretrained [CKPT_FILE] --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
  ```

导出的模型会以模型的结构名字命名并且保存在当前目录下, 注意: FILE_FORMAT 必须在 ["AIR", "MINDIR"]中选择。

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用ImageNet-1k数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

    ```bash
    # Ascend310 inference
    bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
    Top1 acc: 0.72242
    Top5 acc: 0.90734
    ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的deit训练

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型|RepVGG|
| 模型版本              | deit_small_patch16_224                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2022-06-26                                 |
| MindSpore版本          | 1.5.1                                                 |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像                                              |
| 训练参数        | epoch=310, batch_size=1024       |
| 优化器                  | AdamW                                                    |
| 损失函数              | SoftTargetCrossEntropy                                       |
| 损失| 0.87|
| 输出                    | 概率                                                 |
| 分类准确率             | 八卡：top1:80.234% top5:94.984%                   |
| 速度                      | 8卡： 271.135毫秒/步                        |
| 训练耗时          |32h37min38s（run on ModelArts）|

### 推理性能

#### ImageNet-1k上的deit推理

| 参数          | Ascend                          |
| ------------- | ------------------------------- |
| 模型          | Deit                          |
| 模型版本      | deit_small_patch16_224                        |
| 资源          | Ascend 310                      |
| 上传日期      | 2022-03-12                      |
| MindSpore版本 | 1.3.0                           |
| 数据集        | ImageNet-1k Val，共50,000张图像 |
| 分类准确率    | top1:72.242%,top5:90.734%        |
| 速度          | 平均耗时2.90996 每张          |
| 推理耗时      | 约10min                         |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)