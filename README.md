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
export WANDB_API_KEY=X #X = your api key
export WANDB_ENTITY=Y  #Y = your entity
export WANDB_PROJECT=prjectname
python train.py --config src/configs/deit_small_patch16_64.yaml
```

a0f2005d28a0d578c0fb9ec3f9f5e9f32b682848 

