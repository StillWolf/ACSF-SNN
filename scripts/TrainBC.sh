#!/bin/bash

echo "Start a Tmux session."

tmux new -d -s TrainBC_TD3 -n window0

## Split the window to 4 panes.
tmux split-window -h -t TrainBC_TD3:window0
tmux split-window -v -t TrainBC_TD3:window0.0
tmux split-window -v -t TrainBC_TD3:window0.2

## Run the programs sequentially.
tmux send -t TrainBC_TD3:window0.0 "conda deactivate" ENTER
tmux send -t TrainBC_TD3:window0.1 "conda deactivate" ENTER
tmux send -t TrainBC_TD3:window0.2 "conda deactivate" ENTER
tmux send -t TrainBC_TD3:window0.3 "conda deactivate" ENTER

tmux send -t TrainBC_TD3:window0.0 "conda deactivate" ENTER
tmux send -t TrainBC_TD3:window0.1 "conda deactivate" ENTER
tmux send -t TrainBC_TD3:window0.2 "conda deactivate" ENTER
tmux send -t TrainBC_TD3:window0.3 "conda deactivate" ENTER

tmux send -t TrainBC_TD3:window0.0 "conda activate RL" ENTER
tmux send -t TrainBC_TD3:window0.1 "conda activate RL" ENTER
tmux send -t TrainBC_TD3:window0.2 "conda activate RL" ENTER
tmux send -t TrainBC_TD3:window0.3 "conda activate RL" ENTER

tmux send -t TrainBC_TD3:window0.0 "python BehavioralCloning.py --env=Ant-v3 --gpu=0 --seed=9 --buffer=TD3" ENTER
tmux send -t TrainBC_TD3:window0.1 "python BehavioralCloning.py --env=Ant-v3 --gpu=1 --seed=10 --buffer=TD3" ENTER
tmux send -t TrainBC_TD3:window0.2 "python BehavioralCloning.py --env=Ant-v3 --gpu=2 --seed=11 --buffer=TD3" ENTER
tmux send -t TrainBC_TD3:window0.3 "python BehavioralCloning.py --env=Ant-v3 --gpu=2 --seed=12 --buffer=TD3" ENTER

## Attach the Tmux session to the front.
tmux a -t TrainBC_TD3