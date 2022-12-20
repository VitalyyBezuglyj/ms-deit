# ms-deit
Educational implementation of DeiT using the Mindspore framework


## Build && Run Docker

```bash
cd ms-deit/docker

# build container
./build.sh

# start container
./start.sh

# Connect to container
./into.sh
```

## Prepare data

```bash
./into.sh

cd ms-deit/src/data

# Download  & unzip tiny imagenet
./download_tiny_imagenet.sh

# return to ms-deit dir
cd ../..
```

## Train

```bash 
python train.py --config src/configs/deit_small_patch16_64.yaml
```

## TODO:
1. add `WANDB` logging
2. Test &&/|| Fix `eval.py`