#!/bin/bash

export SSH_USERNAME="user"
export SSH_PORT="22"
export SSH_HOST="108.61.192.71"

# sync files
rsync -avz \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='output' \
    --exclude='*.db' \
    --exclude='*.db-wal' \
    --exclude='*.db-shm' \
    --exclude='*.db-journal' \
    -e "ssh -p $SSH_PORT" . $SSH_USERNAME@$SSH_HOST:/home/user/neurallenge_cl

# sleep
sleep 3

# open shell
ssh -p $SSH_PORT $SSH_USERNAME@$SSH_HOST << EOF
# install dependencies
sudo apt install -y xxd opencl-headers ocl-icd-opencl-dev pocl-opencl-icd libsqlite3-dev tmux

# build project
cd /home/user/neurallenge_cl
make clean
make server

# run server
tmux kill-session -t neurallenge 2>/dev/null
tmux new-session -d -s neurallenge './output/server'
EOF
