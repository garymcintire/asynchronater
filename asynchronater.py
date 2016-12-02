import sys
import os
import time
import filelock
import numpy as np
import cPickle as pickle
import keras

'''
    Written by Gary McIntire
    feel free to share
'''

#  from asynchronater import asynchronate, asynchronater_launch
'''
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
each episode. Or it may be better to call it every 10 or so action cycles. Experimentation is
needed here.
Typically the net_list will consist of the policy network but but may also include the value_function, target networks or others.

Requirements:
python
numpy
keras       # tho it could easily be modified. Needs to be able to access the weights and save.

Its useful to pass the num_processes and update_frequency to your RL program for experimenting

'''

last_time = 0
update_secs = 0
last_net_wt_list = []
dir = '/tmp/'   # use /tmp to get the RAM speedup
mdl_file = dir + 'file_asynchronater_mdl_'
g_filename = dir +'file_asynchronater_gs'
output_file='trash1_'       # each process outputs stdout to this file with the my_id appended
my_id = -1
lrr = .001
drag_coeff = .002

def save_mdl(mdl, filename):
    # serialize model to YAML
    # model_yaml = mdl.to_yaml()
    # with open(filename+'.yaml', "w") as yaml_file:
    #     yaml_file.write(model_yaml)
    # serialize weights to HDF5
    mdl.save_weights(filename+".h5", True)
    # print 'my_id',my_id, ' Saved model '+filename +' to disk'

def get_wt_list(net_list):
    return [ [ wts.get_value()  for wts in net.trainable_weights ] for net in net_list ]

def compute_wt_changes(last_net_wt_list, net_list):
    return [ [ wts.get_value() - last_wts for last_wts, wts in zip(last_net_wts, net.trainable_weights) ] for last_net_wts, net in zip(last_net_wt_list, net_list) ]
def RMSprop(netwts, wtchg, gv):
    g = gv[0]
    alpha = .9
    epsilon = 1e-8
    # alpha_v = .9
    g = alpha*g + (1-alpha)*wtchg**2
    delta = wtchg / np.sqrt(g+epsilon)
    # print 'delta_max',  delta.max()
    # drag = 0
    # if drag_coeff > 0:
    #     vel = alpha_v*vel + (1-alpha_v)*delta
    #     unsigned_drag = vel**2
    #     drag = unsigned_drag * np.sign(vel)
    # print 'drag_force', drag.max()
    # delta = np.clip(-.5,.5)
    # print 'dragbool', (drag * (unsigned_drag>5.5))
    # netwts.set_value(netwts.get_value() + lrr*delta  - drag_coeff*(drag * (unsigned_drag>1.5)))
    netwts.set_value(netwts.get_value() + lrr*delta )
    # return [g, vel]
    return [g, []]
def add_wt_changes(wt_changes, net_list):
    for i, mdl in zip(xrange(len(net_list)), net_list):
        mdl.load_weights(mdl_file+str(i)+'.h5')        # load file weights to our current nets
    with open(g_filename, 'r') as fp:
        gss = pickle.load(fp)
    # [ [ netwts.set_value(netwts.get_value() + wtchg)  for wtchg, netwts in zip(wt_list, net.trainable_weights) ] for wt_list, net in zip(wt_changes, net_list) ]  # add in the wt changes to global models
    return [ [ RMSprop(netwts, wtchg, g)  for wtchg, netwts, g in zip(wt_list, net.trainable_weights, gs) ] for wt_list, net, gs in zip(wt_changes, net_list, gss) ]  # add in the wt changes to global models

# use this to synchronize some Keras neural nets across several processes that are launched by asynchronater_launcher
# Do this periodically, usually after making wt updates. Too often doesn't work well. Too infrequent allows the nets to drift
# too far from each other.
def asynchronate(net_list, use_wt_files=False):
    global last_time
    global update_secs
    global last_net_wt_list
    global my_id
    now = time.time()
    if last_time == 0:  # first time thru?
        args = sys.argv
        for i in xrange(len(args)):
            if args[i] == '-update_secs': update_secs = float(args[i+1])
    # print 'last_time', last_time, 'update_secs', update_secs,  'sum', last_time + update_secs, 'time now',now, 'truth', last_time + update_secs < now
    if last_time + update_secs > now: return tuple(net_list)   # do nothing if not time
    last_time = now
    lock = filelock.FileLock("asynchronater_lock_file")
    with lock:
        if not os.path.exists(mdl_file+str(0)+'.h5'):
            print 'my_id', my_id, 'saving the initial model'
            for e,i in zip(net_list, xrange(len(net_list))): save_mdl(e, mdl_file+str(i))   # if we are first, save our wts to global file
            # gs = [ [ [np.zeros_like(wts.get_value()),np.zeros_like(wts.get_value())]  for wts in net.trainable_weights ] for net in net_list ]
            gs = [ [ [np.zeros_like(wts.get_value())]  for wts in net.trainable_weights ] for net in net_list ]
            # vel = [ [ np.zeros_like(wts.get_value())  for wts in net.trainable_weights ] for net in net_list ]
            with open(g_filename, 'w') as fp:
                pickle.dump(gs, fp)
        else:
            print 'my_id',my_id,'is updating its weights from the global file'
            wt_changes = compute_wt_changes(last_net_wt_list, net_list)
            # wtchg = np.asarray(wt_changes)
            # print 'wt_changes',wtchg.min(),wtchg.mean(),wtchg.max()
            # print 'wtchgs',wt_changes
            gs = add_wt_changes(wt_changes, net_list)        # add to the net loaded from file and write to original net
            if gs:
                with open(g_filename, 'w') as fp:
                    pickle.dump(gs , fp)
            for e,i in zip(net_list, xrange(len(net_list))): save_mdl(e, mdl_file+str(i))   # save these updated original nets
    last_net_wt_list = get_wt_list(net_list)

def asynchronater_launch(num_processes, lr=.003):
    global my_id
    global update_secs
    global lrr
    if num_processes <= 1: return
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    lrr = lr
    args = sys.argv
    # if '-update_secs' not in args:
    #     sys.exit('YOU MUST SUPPLY   -update_secs num    FROM THE COMMAND LINE')
    if '-my_id' in args:
        for i in xrange(len(args)):
            if args[i] == '-my_id': my_id = int(args[i+1])
    if '-dont_recurse' in args:
        print 'found dont_recurse so returning and running program. my_id',my_id
        return
    # first process my_id -1 runs code below
    os.system('rm asynchronater_lock_file')
    lock = filelock.FileLock("asynchronater_lock_file")
    with lock:
        print 'asynchronater_launch is deleting the '+mdl_file+'.yaml and .h5 files'
        os.system('rm '+mdl_file+'*.yaml')
        os.system('rm '+mdl_file+'*.h5')
        os.system('rm '+g_filename)
    redirect = ''
    for i in xrange(num_processes):
        if i > 0: redirect = '>'+output_file+str(i)
        cmd = 'python ' + ' '.join(sys.argv) + ' -my_id '+str(i) + ' -dont_recurse '+redirect+'&'
        print 'my_id',my_id,'is starting id',i,'------------- with',cmd
        os.system(cmd)
        print 'my_id',my_id,'returned from starting id',i,'------------- with',cmd
    sys.exit('my_id '+str(my_id)+' is exiting')