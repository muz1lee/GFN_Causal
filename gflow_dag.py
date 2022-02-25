import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count
import numpy.random as rd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import random
import time

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation,DAG
from castle.algorithms.gradient import corl
from sklearn import preprocessing
import os
import logging
import argparse
import platform
import random
from tqdm import tqdm
import numpy as np
import torch
from castle.common import BaseLearner, Tensor, consts
from castle.algorithms.gradient.gflownet.torch.frame import Reward
from castle.algorithms.gradient.gflownet.torch.utils.data_loader import DataGenerator
from castle.algorithms.gradient.gflownet.torch.utils.graph_analysis import get_graph_from_order, pruning_by_coef
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# import gym

random.seed(1022)

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/flownet_back.gz', type=str)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')


parser.add_argument("--method", default='flownet', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=4, help="Minibatch size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=3000, type=int)
parser.add_argument("--num_empirical_loss", default=200000, type=int,
                    help="Number of samples used to compute the empirical distribution loss")
# Env
# parser.add_argument('--func', default='cal_reward_simple')

parser.add_argument("--n_node", default=3, type=int)

# MCMC
parser.add_argument("--bufsize", default=4, help="MCMC buffer size", type=int)
parser.add_argument("--is_mcmc", default=False)

# Flownet
parser.add_argument("--bootstrap_tau", default=0., type=float)
parser.add_argument("--replay_strategy", default='none', type=str) # top_k none
parser.add_argument("--replay_sample_size", default=2, type=int)
parser.add_argument("--replay_buf_size", default=100, type=float)

# PPO
parser.add_argument("--ppo_num_epochs", default=32, type=int) # number of SGD steps per epoch
parser.add_argument("--ppo_epoch_size", default=1, type=int) # number of sampled minibatches per epoch
parser.add_argument("--ppo_clip", default=0.2, type=float)
parser.add_argument("--ppo_entropy_coef", default=1e-1, type=float)
parser.add_argument("--clip_grad_norm", default=0., type=float)

# SAC
parser.add_argument("--sac_alpha", default=0.98*np.log(1/3), type=float)

#DQN
parser.add_argument("--explore_rate",default= 0.9,type=float)
parser.add_argument("--reward_decay",default= 0.9,type=float)



_dev = [torch.device('cpu')]
tf = lambda x: torch.Tensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

# data
weighted_random_dag = DAG.erdos_renyi(n_nodes=3,n_edges=4,weight_range=(0.5,2.0),seed=1)
datasets = IIDSimulation(weighted_random_dag,n=50,method='linear',sem_type = 'gauss')
true_causal_matrix,X = datasets.B,datasets.X
n_samples = X.shape[0]
seq_length = X.shape[1]
data_generator = DataGenerator(dataset=X,normalize=False)
input_batch = data_generator.draw_batch(batch_size=5,dimension=100)

# Reward
reward =Reward(input_data=data_generator.dataset.cpu().detach().numpy(),
                    reward_mode='episodic',
                    score_type='BIC',
                    regression_type='LR',
                    alpha=1.0)

def set_device(dev):
    _dev[0] = dev

from castle.algorithms.gradient.corl.torch.frame import Reward


def cal_reward_simple(potential_matrix):
    potential_matrix = potential_matrix * (1-np.eye(potential_matrix.shape[0]))
    RSS_ls = []
    for i in range(3):
        RSSi = reward.cal_RSSi(i,potential_matrix)
        RSS_ls.append(RSSi)

    RSS_ls = np.array(RSS_ls)

    BIC = np.log(np.sum(RSS_ls) / 50 + 1e-8)

    return BIC

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(
        *(sum
        ( [
              [nn.Linear(i, o)] +

              ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate  (  zip  (l, l[1:])  )

          ],
              [])

          + tail))
def make_opt(params, args):
    params = list(params)
    if not len(params):
        return None
    if args.opt == 'adam':
        opt = torch.optim.Adam(params, args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2))
    elif args.opt == 'msgd':
        opt = torch.optim.SGD(params, args.learning_rate, momentum=args.momentum)
    return opt



class CausalEnv:

    def __init__(self,n_node):
        self.n_node = n_node
        #self.allow_backward = allow_backward  # If true then this is a MCMC ergodic env, otherwise a DAG

    def parent_transitions(self,s):
            parents = []
            actions = []
            results = np.where(s==1)


            # if len(results[0])==0:
            #     parents.append(s)
            #     actions.append(-1)
            #     return s , actions

            for item in results[0]:
                sp = s+0
                sp[item] -= 1

                parents.append(sp)
                actions.append(item)

            return parents, actions

    def step(self, a,s):
        # _s = s
        # s = (tf(np.zeros((self.n_node * self.n_node))) if s is None else s) + 0
        # s = (self._state if s is None else s) + 0
        # sample 没超过维度且需要没有选中重复的
        # if a < len(s) and s[int(a)]==0:
        no_step = s[int(a)] != 0
        if s[int(a)] == 0:
            new_s = s + 0
            new_s[int(a)] = 1

            matrix_new_s = np.array(new_s).reshape(args.n_node,args.n_node)

            row_ix = int(a) // args.n_node
            # print('row_ix',row_ix)
            col_ix = int(a) % args.n_node  # 3 is dim
            # print('col_ix',col_ix)
            Trans_ix = col_ix * args.n_node + row_ix

            # print('Trans_ix',Trans_ix)
            # 看加了该元素时，是否让对角线为1, 看是否让对称的元素重复为1,如果是则说明结束了（不能走）
            # if len(np.where(np.diag(matrix_new_s)==1)[0]) >0 or new_s[Trans_ix] == 1:

            # 看是否让对称的元素重复为1 且 不是让对角线为1 （在这里我们认定对角线还是可以走）
            no_step = new_s[Trans_ix] == 1 and len(np.where(np.diag(matrix_new_s)==1)[0]) == 0

        #
        # else:
        #     # sample超过维度了
        #     no_step = True

        # if _s is None:
        #     self._state = s
        #     self._step += 1

        return np.array(s) if no_step else np.array(new_s), cal_reward_simple(
            np.array(s).reshape(args.n_node, args.n_node)) if no_step else 0, torch.Tensor([no_step])

class ReplayBuffer:
    def __init__(self, args, env):
        self.buf = []
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = args.replay_buf_size
        self.env = env

    def add(self, x, r_x):
        if self.strat == 'top_k':
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, x)])[-self.bufsize:]

    def sample(self):
        if not len(self.buf):
            return []
        idxs = np.random.randint(0, len(self.buf), self.sample_size)
        return sum([self.generate_backward(*self.buf[i]) for i in idxs], [])

    def generate_backward(self, r, s0):
        s = np.int8(s0)
        # os0 = self.env.obs(s)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        #used_stop_action = s.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0:
            #parents, actions = self.env.parent_transitions(s, used_stop_action)

            parents, actions = self.env.parent_transitions(s)
            # add the transition
            traj.append([tf(i) for i in (parents, actions, [r], [done])])
            # Then randomly choose a parent state

            i = np.random.randint(0, len(parents))
            a = actions[i]
            s[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj

class FlowNetAgent:
    def __init__(self, args, envs):
        self.n_node = args.n_node
        self.model = make_mlp([args.n_node*args.n_node ] +
                    [args.n_hid] * args.n_layers + [args.n_node*args.n_node])

        self.model.to(args.dev)
        self.target = copy.deepcopy(self.model)
        self.envs = envs
        self.tau = args.bootstrap_tau
        self.replay = ReplayBuffer(args, envs[0])

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited):
        batch = []
        batch += self.replay.sample()

        s = tf([np.zeros((self.n_node * self.n_node)) for i in self.envs])
        done = [False] * mbsize
        while not all(done):

            with torch.no_grad():
                model_outcome = self.model(tf(s))
                acts = Categorical(logits=model_outcome).sample()

            # for i,a in zip([env for done, env in zip(done, self.envs) if not done], acts):
            #     step = i.step(a)
            
            step = [i.step(a,state) for i,a,state in zip([e for d, e in zip(done, self.envs) if not d], acts ,s )]

            p_a = [self.envs[0].parent_transitions(sp) for a, (sp, r, done) in zip(acts, step)]

            #
            # batch += [[ i for i in (tf([np.array(p)]), tf(a), tf([r]), tf([sp]), tf([d]))]
            #           for (p, a), (sp, r, d) in zip(p_a, step)]

            batch += [[tf(i) for i in (p, a, [r], [sp], [d])]
                      for (p, a), (sp, r, d) in zip(p_a, step)]
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf(np.array([i[0] for i in step if not i[2]]))


            for (sp, r, d) in step:
                if d:
                    all_visited.append(tuple(sp))
                    self.replay.add(tuple(sp), r)

        return batch

    def learn_from(self, it, batch):

        loginf = torch.Tensor([1000])

        # batch_result = batch


        batch_idxs = tl(sum([[i]*len(parents) for i, (parents,_,_,_,_) in enumerate(batch)], []))

        parents, actions, r, sp, done = map(torch.cat, zip(*batch))

        parents_Qsa = self.model(torch.tensor(parents))  [torch.arange(parents.shape[0]), actions.long()]

        in_flow = torch.log(torch.zeros((sp.shape[0],))
                            .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))

        next_q = self.model(torch.tensor(sp))
        next_qd = next_q * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        out_flow = torch.logsumexp(   torch.cat(  [torch.log( r)[:, None], next_qd  ], 1)  , 1)

        loss = (in_flow - out_flow).pow(2).mean() # explosion

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((in_flow - out_flow) * (1-done)).pow(2).sum() / ((1-done).sum() + 1e-20)

        return loss, term_loss, flow_loss


def main(args):
    weighted_random_dag = DAG.erdos_renyi(n_nodes=args.n_node,n_edges=4,weight_range=(0.5,2.0),seed=1)
    # weighted_random_dag
    datasets = IIDSimulation(weighted_random_dag,n=50,method='linear',sem_type = 'gauss')
    # datasets
    true_causal_matrix,X = datasets.B,datasets.X

    data_generator = DataGenerator(dataset=X,normalize=False)

    input_batch = data_generator.draw_batch(batch_size=5,dimension=100)

    args.dev = torch.device(args.device)
    set_device(args.dev)


    reward =Reward(input_data=data_generator.dataset.cpu().detach().numpy(),
                    reward_mode='dense',
                    score_type='BIC',
                    regression_type='LR',
                    alpha=1.0)

    env = CausalEnv(args.n_node)
    envs = [CausalEnv(args.n_node) for i in range(args.bufsize)]

    agent = FlowNetAgent(args, envs)

    opt = make_opt(agent.parameters(), args)

    # metrics
    all_losses = []
    all_visited = []
    # empirical_distrib_losses = []

    ttsr = max(int(args.train_to_sample_ratio), 1) # 1
    sttr = max(int(1/args.train_to_sample_ratio), 1) # sample to train ratio

    start_time = time.time()
    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        data = []

        for j in range(sttr):

            data += agent.sample_many(args.mbsize, all_visited) # mbsize = 16


        for j in range(ttsr):
            losses = agent.learn_from(i * ttsr + j, data) # returns (opt loss, *metrics)
            if losses is not None:
                print('losses[0]',losses[0])
                with torch.autograd.detect_anomaly():
                    torch.Tensor(losses[0]).backward()
                # losses[0] tensor(inf, grad_fn=<MeanBackward0>)

                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                                   args.clip_grad_norm)
                opt.step()
                opt.zero_grad()
                all_losses.append([i.item() for i in losses])
                all_losses.append(losses)


    #     if not i % 100:
    #         print('i',i)
    #         empirical_distrib_losses.append(
    #             compute_empirical_distribution_error(env, all_visited[-args.num_empirical_loss:]))
    #         if args.progress:
    #             k1, kl = empirical_distrib_losses[-1]
    #             print('empirical L1 distance', k1, 'KL', kl)
    #             if len(all_losses):
    #                 print(*[f'{np.mean([i[j] for i in all_losses[-100:]]):.5f}'
    #                         for j in range(len(all_losses[0]))])
    end_time = time.time()
    time_last =  end_time-start_time
    print(' -----------------Complete the whole training steps -----------------')
    print('Time_last：',time_last)
    print('agent.model',agent.model)
    print('agent.parameters()',agent.parameters())

    print(('ori_result'),agent.model(tf(np.zeros((args.n_node*args.n_node)))) )

    new_result = agent.model(tf(np.zeros((args.n_node*args.n_node)))).detach().numpy().reshape(1, -1)
    print('new_result',new_result)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(new_result.reshape(-1,1))
    print('x_minmax',x_minmax)

    threshold =   np.sort(x_minmax.reshape(-1))[-3]
    mask = x_minmax>=threshold
    mask = mask.astype(np.int64)
    print('pred_matrix',mask.reshape(args.n_node,args.n_node))
   # print(  torch.nn.functional.log_softmax(new_result,dim=1))
    print('true_causal_matrix',true_causal_matrix)
    # root = os.path.split(args.save_path)[0]
    # os.makedirs(root, exist_ok=True)
    # pickle.dump(
    #     {'losses':  all_losses,       #np.float32(all_losses),
    #      #'model': agent.model.to('cpu') if agent.model else None,
    #      'params': [i.data.to('cpu').numpy() for i in agent.parameters()],
    #      'visited': np.int8(all_visited),
    #      'emp_dist_loss': empirical_distrib_losses,
    #      'true_d': env.true_density()[0],
    #      'args':args,
    #      'time':time_last},
    #     gzip.open(args.save_path, 'wb'))

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
