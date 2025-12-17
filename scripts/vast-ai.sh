#!/bin/bash

export SSH_USERNAME="root"
export SSH_PORT="11357"
export SSH_HOST="70.69.213.236"

# sync files
rsync -avz --exclude='.git' --exclude='.DS_Store' --exclude='output' -e "ssh -p $SSH_PORT" . $SSH_USERNAME@$SSH_HOST:/workspace/neurallenge_cl

# sleep
sleep 3

# open shell
ssh -p $SSH_PORT $SSH_USERNAME@$SSH_HOST << EOF
# kill
killall miner

# install dependencies
apt install -y xxd opencl-headers ocl-icd-opencl-dev pocl-opencl-icd libsqlite3-dev

# build project
cd /workspace/neurallenge_cl
make clean
make GLOBAL_SIZE=131072 LOCAL_SIZE=128 HASHES_PER_THREAD=128

# run
./output/miner
EOF
