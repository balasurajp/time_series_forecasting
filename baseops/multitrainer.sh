#!/bin/bash

command=$1
interpreter="~/myData/envs/py36/bin/activate"

if [ "$command" = "start" ]
then
	tmux new -d -s ptaccu
	tmux send-keys -t ptaccu.0 "source $interpreter" ENTER
	tmux send-keys -t ptaccu.0 "python baseops/trainmodule.py -alg transformer -cfg ptaccu_transformer -sst 1 -gpu 0,1" ENTER
	tmux send-keys -t ptaccu.0 "python baseops/testmodule.py -alg transformer -cfg ptaccu_transformer -gpu 0,1" ENTER

	tmux new -d -s ptacnd
	tmux send-keys -t ptacnd.0 "source $interpreter" ENTER
	tmux send-keys -t ptacnd.0 "python baseops/trainmodule.py -alg transformer -cfg ptacnd_transformer -sst 1 -gpu 2,3" ENTER
	tmux send-keys -t ptacnd.0 "python baseops/testmodule.py -alg transformer -cfg ptacnd_transformer -gpu 2,3" ENTER

elif [ "$command" = "stop" ]
then
	tmux kill-window -t ptaccu
	tmux kill-window -t ptacnd

else
  echo "No command parsed!"
fi
