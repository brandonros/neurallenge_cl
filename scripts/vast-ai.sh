#!/bin/bash

export SSH_USERNAME="root"
export SSH_PORT="61108"
export SSH_HOST="70.69.213.236"

# sync files
rsync -avz \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='output' \
    --exclude='*.db' \
    --exclude='*.db-wal' \
    --exclude='*.db-shm' \
    --exclude='*.db-journal' \
    -e "ssh -p $SSH_PORT" . $SSH_USERNAME@$SSH_HOST:/workspace/neurallenge_cl

# sleep
sleep 3

# setup and build and start
ssh -p $SSH_PORT $SSH_USERNAME@$SSH_HOST << EOF
# install dependencies
apt install -y xxd tmux

# build project
cd /workspace/neurallenge_cl
make clean
make miner GLOBAL_SIZE=131072 LOCAL_SIZE=128 HASHES_PER_THREAD=128

# run in tmux
tmux kill-session -t neurallenge 2>/dev/null
tmux new-session -d -s neurallenge './output/miner -u anonymous -s http://108.61.192.71:8080 -b 30'
EOF

# attach to tmux session to wath
ssh -t -p $SSH_PORT $SSH_USERNAME@$SSH_HOST "tmux attach -t neurallenge"
