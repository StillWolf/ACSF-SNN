#!/bin/bash

echo "Start a Tmux session."

tmux new -d -s TrainDDPG -n window0

## Split the window to 4 panes.
tmux split-window -h -t TrainDDPG:window0
tmux split-window -v -t TrainDDPG:window0.0
tmux split-window -v -t TrainDDPG:window0.2

## Run the ROS programs sequentially.
tmux send -t TrainDDPG:window0.0 "conda deactivate" ENTER
tmux send -t TrainDDPG:window0.1 "conda deactivate" ENTER
tmux send -t TrainDDPG:window0.2 "conda deactivate" ENTER
tmux send -t TrainDDPG:window0.3 "conda deactivate" ENTER

tmux send -t TrainDDPG:window0.0 "conda deactivate" ENTER
tmux send -t TrainDDPG:window0.1 "conda deactivate" ENTER
tmux send -t TrainDDPG:window0.2 "conda deactivate" ENTER
tmux send -t TrainDDPG:window0.3 "conda deactivate" ENTER

tmux send -t TrainDDPG:window0.0 "conda activate RL" ENTER
tmux send -t TrainDDPG:window0.1 "conda activate RL" ENTER
tmux send -t TrainDDPG:window0.2 "conda activate RL" ENTER
tmux send -t TrainDDPG:window0.3 "conda activate RL" ENTER

tmux send -t TrainDDPG:window0.0 "python main.py --env=Hopper-v3 --gpu=0 --seed=5 --train_behavioral --mode=DDPG" ENTER
tmux send -t TrainDDPG:window0.1 "python main.py --env=Hopper-v3 --gpu=2 --seed=6 --train_behavioral --mode=DDPG" ENTER
tmux send -t TrainDDPG:window0.2 "python main.py --env=Hopper-v3 --gpu=0 --seed=7 --train_behavioral --mode=DDPG" ENTER
tmux send -t TrainDDPG:window0.3 "python main.py --env=Hopper-v3 --gpu=2 --seed=8 --train_behavioral --mode=DDPG" ENTER

## Attach the Tmux session to the front.
tmux a -t TrainDDPG