#!/bin/bash

command=$1
interpreter="~/myData/envs/py36/bin/activate"

if [ "$command" = "start" ]
then
	tmux new -d -s ptacnd
	tmux send-keys -t ptacnd.0 "source $interpreter" ENTER
	tmux send-keys -t ptacnd.0 "python baseops/trainmodule.py -alg seq2seq -cfg ptacnc_seq2seq -sst 1 -gpu 0" ENTER
	tmux send-keys -t ptacnd.0 "python baseops/trainmodule.py -alg seq2seq -cfg ptacnp_seq2seq -sst 1 -gpu 0" ENTER
	tmux send-keys -t ptacnd.0 "python baseops/testmodule.py -alg seq2seq -cfg ptacnc_seq2seq -gpu 0" ENTER
	tmux send-keys -t ptacnd.0 "python baseops/testmodule.py -alg seq2seq -cfg ptacnp_seq2seq -gpu 0" ENTER

	tmux new -d -s ptaccu_home
	tmux send-keys -t ptaccu_home.0 "source $interpreter" ENTER
	tmux send-keys -t ptaccu_home.0 "python baseops/trainmodule.py -alg seq2seq -cfg ptaccu_home_seq2seq -sst 1 -gpu 1" ENTER
	tmux send-keys -t ptaccu_home.0 "python baseops/testmodule.py -alg seq2seq -cfg ptaccu_home_seq2seq -gpu 1" ENTER

	tmux new -d -s ptaccu_office
	tmux send-keys -t ptaccu_office.0 "source $interpreter" ENTER
	tmux send-keys -t ptaccu_office.0 "python baseops/trainmodule.py -alg seq2seq -cfg ptaccu_office_seq2seq -sst 2 -gpu 2" ENTER
	tmux send-keys -t ptaccu_office.0 "python baseops/testmodule.py -alg seq2seq -cfg ptaccu_office_seq2seq -gpu 2" ENTER

	tmux new -d -s ptaccu_village
	tmux send-keys -t ptaccu_village.0 "source $interpreter" ENTER
	tmux send-keys -t ptaccu_village.0 "python baseops/trainmodule.py -alg seq2seq -cfg ptaccu_village_seq2seq -sst 3 -gpu 3" ENTER
	tmux send-keys -t ptaccu_village.0 "python baseops/testmodule.py -alg seq2seq -cfg ptaccu_village_seq2seq -gpu 3" ENTER

elif [ "$command" = "stop" ]
then
	tmux kill-window -t ptacnd
	tmux kill-window -t ptaccu_home
	tmux kill-window -t ptaccu_office
	tmux kill-window -t ptaccu_village

else
  echo "No command parsed!"
fi
