# asynchronater
Code for parallelizing your reinforcement learning algorithm


This is for running multiple reinforcement learning algorithms simultaneously in separate processes. They
will synchronize the weights of their networks when asynchronate is called.

If you have an RL algorithm written and you want to parallelize it while sharing weight updates in the manner of
Deepmind's A3C algorithm, use this.

Their are only two functions exported here...

asynchronater_launch(num_processes)     # start several processes

asynchronate(net_list)                  # add my weight changes to the global wt file and update my nets with the result

Near the beginning of your RL code, call asynchronater_launch(num_processes). This
uses the command line arguments to launch multiple processes.

Somewhere in the RL algorithm call  asynchronate(net_list). It may be useful to call it
each episode. Or it may be better to call it every 10(or even 1) or so action cycles. Experimentation is
needed here.
Typically the net_list will consist of the policy network but may also include the value_function, target networks or others.
Requirements:
python
numpy
keras       # tho it could easily be modified. Needs to be able to access the weights and save.

Its useful to pass the num_processes and update_frequency to your RL program for experimenting

