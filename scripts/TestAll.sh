#!/bin/bash

echo "Start a Tmux session."

## Create a Tmux session "TestModel" in a window "window0" started in the background.
tmux new -d -s TestModel -n window0

## Split the window to 4 panes.
tmux split-window -h -t TestModel:window0
tmux split-window -v -t TestModel:window0.0
tmux split-window -v -t TestModel:window0.2

## Run the programs sequentially.
tmux send -t TestModel:window0.0 "conda deactivate" ENTER
tmux send -t TestModel:window0.1 "conda deactivate" ENTER
tmux send -t TestModel:window0.2 "conda deactivate" ENTER
tmux send -t TestModel:window0.3 "conda deactivate" ENTER

tmux send -t TestModel:window0.0 "conda deactivate" ENTER
tmux send -t TestModel:window0.1 "conda deactivate" ENTER
tmux send -t TestModel:window0.2 "conda deactivate" ENTER
tmux send -t TestModel:window0.3 "conda deactivate" ENTER

tmux send -t TestModel:window0.0 "conda activate RL" ENTER
tmux send -t TestModel:window0.1 "conda activate RL" ENTER
tmux send -t TestModel:window0.2 "conda activate RL" ENTER
tmux send -t TestModel:window0.3 "conda activate RL" ENTER

tmux send -t TestModel:window0.0 "python TestModel.py --env=Ant-v3 --gpu=0 --seed=9853" ENTER
tmux send -t TestModel:window0.1 "python TestModel.py --env=HalfCheetah-v3 --gpu=1 --seed=9853" ENTER
tmux send -t TestModel:window0.2 "python TestModel.py --env=Walker2d-v3 --gpu=0 --seed=9853" ENTER
tmux send -t TestModel:window0.3 "python TestModel.py --env=Hopper-v3 --gpu=1 --seed=9853" ENTER

## Attach the Tmux session to the front.
tmux a -t TestModel


