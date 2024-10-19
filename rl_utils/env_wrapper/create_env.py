# from rl_utils.env_wrapper.atari_wrapper import make_atari, wrap_deepmind
from rl_utils.logger import logger, bench
from rl_utils.env_wrapper.blob_env import BlobEnv
import os
# import gym

"""
this functions is to create the environments

"""

def create_single_env(args, rank=0):
    # setup the log files
    if rank == 0:
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        log_path = args.log_dir + '/{}/'.format(args.env_name)
        logger.configure(log_path)
    
    env = BlobEnv()
    env = bench.Monitor(env, logger.get_dir())
    
    # # start to create environment
    # if args.env_type == 'atari':
    #     # create the environment
    #     env = make_atari(args.env_name)
    #     # the monitor
    #     env = bench.Monitor(env, logger.get_dir())
    #     # use the deepmind environment wrapper
    #     env = wrap_deepmind(env, frame_stack=True)
    # else:
    #     env = gym.make(args.env_name)
    #     # add log information
    #     env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    return env
