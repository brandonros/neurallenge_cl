#!/bin/bash

export SSH_USERNAME="root"
export SSH_PORT="27651"
export SSH_HOST="74.48.140.178"

# sync files
rsync -avz --exclude='.git' --exclude='.DS_Store' --exclude='output' -e "ssh -p $SSH_PORT" . $SSH_USERNAME@$SSH_HOST:/workspace/neurallenge_cl

# open shell
ssh -p $SSH_PORT $SSH_USERNAME@$SSH_HOST << EOF
# kill
killall neurallenge

# install dependencies
apt-get install -y xxd

# build project
cd /workspace/neurallenge_cl
make clean
make

# run
./output/neurallenge
EOF
