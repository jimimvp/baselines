#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import roboschool_arg_parser
from baselines import bench, logger
import roboschool

    


def main():
    parser = roboschool_arg_parser()
    parser.add_argument("--nsteps", help="Number of steps for advantage estimation", type=int, default=2048)
    parser.add_argument("--noptsteps", help="Number of optimisation steps", default=10, type=int)
    parser.add_argument("--entcoef", help="Entropy coefficient", default=0.0, type=float)
    parser.add_argument("--mbatches", help="Number of minibatches", default=32, type=int)
    parser.add_argument("--horizon", help="Time horizon", default=1000000, type=int)
    parser.add_argument("--lmda", help="Lambda", default=0.95, type=float)
    parser.add_argument("--gamma", help="Reward discount factor", default=0.99, type=float)


    args = parser.parse_args()
    logger.configure()

    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(args.env)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(args.seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=args.nsteps, nminibatches=args.mbatches,
        lam=args.lmda, gamma=args.gamma, noptepochs=args.noptsteps, log_interval=1,
        ent_coef=args.entcoef,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=args.horizon)


if __name__ == '__main__':
    main()
