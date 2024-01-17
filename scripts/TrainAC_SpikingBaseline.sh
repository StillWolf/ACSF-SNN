#!/bin/bash

echo "Start a Tmux session."

tmux new -d -s TrainSpiking -n window0

## Split the window to 4 panes.
tmux split-window -h -t TrainSpiking:window0
tmux split-window -v -t TrainSpiking:window0.0
tmux split-window -v -t TrainSpiking:window0.2

## Run the programs sequentially.
tmux send -t TrainSpiking:window0.0 "conda deactivate" ENTER
tmux send -t TrainSpiking:window0.1 "conda deactivate" ENTER
tmux send -t TrainSpiking:window0.2 "conda deactivate" ENTER
tmux send -t TrainSpiking:window0.3 "conda deactivate" ENTER

tmux send -t TrainSpiking:window0.0 "conda deactivate" ENTER
tmux send -t TrainSpiking:window0.1 "conda deactivate" ENTER
tmux send -t TrainSpiking:window0.2 "conda deactivate" ENTER
tmux send -t TrainSpiking:window0.3 "conda deactivate" ENTER

tmux send -t TrainSpiking:window0.0 "conda activate RL" ENTER
tmux send -t TrainSpiking:window0.1 "conda activate RL" ENTER
tmux send -t TrainSpiking:window0.2 "conda activate RL" ENTER
tmux send -t TrainSpiking:window0.3 "conda activate RL" ENTER

tmux send -t TrainSpiking:window0.0 "python main.py --env=Ant-v3 --gpu=0 --seed=9853 --mode=Spiking --T=8 --buffer=TD3" ENTER
tmux send -t TrainSpiking:window0.1 "python main.py --env=HalfCheetah-v3 --gpu=1 --seed=9853 --mode=Spiking --T=8" ENTER
tmux send -t TrainSpiking:window0.2 "python main.py --env=Walker2d-v3 --gpu=2 --seed=9853 --mode=Spiking --T=8" ENTER
tmux send -t TrainSpiking:window0.3 "python main.py --env=Hopper-v3 --gpu=1 --seed=9853 --mode=Spiking --T=8" ENTER

## Attach the Tmux session to the front.
tmux a -t TrainSpiking