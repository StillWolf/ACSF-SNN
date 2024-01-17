#!/bin/bash

echo "Start a Tmux session."

tmux new -d -s TrainBCQ -n window0

## Split the window to 4 panes.
tmux split-window -h -t TrainBCQ:window0
tmux split-window -v -t TrainBCQ:window0.0
tmux split-window -v -t TrainBCQ:window0.2

## Run the programs sequentially.
tmux send -t TrainBCQ:window0.0 "conda deactivate" ENTER
tmux send -t TrainBCQ:window0.1 "conda deactivate" ENTER
tmux send -t TrainBCQ:window0.2 "conda deactivate" ENTER
tmux send -t TrainBCQ:window0.3 "conda deactivate" ENTER

tmux send -t TrainBCQ:window0.0 "conda deactivate" ENTER
tmux send -t TrainBCQ:window0.1 "conda deactivate" ENTER
tmux send -t TrainBCQ:window0.2 "conda deactivate" ENTER
tmux send -t TrainBCQ:window0.3 "conda deactivate" ENTER

tmux send -t TrainBCQ:window0.0 "conda activate RL" ENTER
tmux send -t TrainBCQ:window0.1 "conda activate RL" ENTER
tmux send -t TrainBCQ:window0.2 "conda activate RL" ENTER
tmux send -t TrainBCQ:window0.3 "conda activate RL" ENTER

tmux send -t TrainBCQ:window0.0 "python main.py --env=Ant-v3 --gpu=0 --seed=4 --mode=BCQ --buffer=TD3" ENTER
tmux send -t TrainBCQ:window0.1 "python main.py --env=HalfCheetah-v3 --gpu=0 --seed=4 --mode=BCQ --buffer=DDPG_9853" ENTER
tmux send -t TrainBCQ:window0.2 "python main.py --env=Walker2d-v3 --gpu=0 --seed=4 --mode=BCQ --buffer=DDPG_9853" ENTER
tmux send -t TrainBCQ:window0.3 "python main.py --env=Hopper-v3 --gpu=0 --seed=4 --mode=BCQ --buffer=DDPG_9853" ENTER

## Attach the Tmux session to the front.
tmux a -t TrainBCQ