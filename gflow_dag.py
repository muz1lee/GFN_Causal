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
parser.add_argument("--n_train_steps", default=2, type=int)
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

weighted_random_dag = DAG.erdos_renyi(n_nodes=3,n_edges=4,weight_range=(0.5,2.0),seed=1)
# weighted_random_dag
datasets = IIDSimulation(weighted_random_dag,n=50,method='linear',sem_type = 'gauss')
# datasets

true_causal_matrix,X = datasets.B,datasets.X
#true_causal_matrix

# X = Tensor(X)
n_samples = X.shape[0]
seq_length = X.shape[1]

# print('n_samples',n_samples)
# print('seq_length',seq_length)

data_generator = DataGenerator(dataset=X,normalize=False)
data_generator
input_batch = data_generator.draw_batch(batch_size=5,dimension=100)
input_batch.shape

reward =Reward(input_data=data_generator.dataset.cpu().detach().numpy(),
                    reward_mode='dense',
                    score_type='BIC',
                    regression_type='LR',
                    alpha=1.0)

def set_device(dev):
    _dev[0] = dev

from castle.algorithms.gradient.corl.torch.frame import Reward


def cal_reward_simple(potential_matrix):
    RSS_ls = []
    for i in range(3):
        RSSi = reward.cal_RSSi(i,potential_matrix)
        RSS_ls.append(RSSi)

    RSS_ls = np.array(RSS_ls)

    BIC = np.log(np.sum(RSS_ls) / 50 + 1e-8)

    return BIC



class CausalEnv:

    def __init__(self,n_node):

        self.n_node = n_node

        #self.allow_backward = allow_backward  # If true then this is a MCMC ergodic env, otherwise a DAG


    # def obs(self, s=None):
    #     s = np.int32(self._state if s is None else s)
    #     z = np.zeros((self.n_node * self.n_node))
    #     z[np.arange(len(s)) * self.horizon + s] = 1
    #     return z

    # def reset(self):
    #     self._state = np.int32([0] * self.ndim)
    #     self._step = 0
    #     return self.obs(), cal_reward_simple(np.array(self._state).reshape(3,3)) , self._state


    # def reset(self):
    #     self._state = np.int32([0] * (self.n_node* self.n_node))
    #     self._step = 0
    #     return self.obs(), self.func(self.s2x(self._state)), self._state

    # def parent_transitions(self, s, used_stop_action):

    #     if used_stop_action:
    #         return [self.obs(s)], [self.ndim]
    #     parents = []
    #     actions = []
    #     for i in range(self.ndim):
    #         if s[i] > 0:
    #             sp = s + 0
    #             sp[i] -= 1
    #             if sp.max() == self.horizon-1: # can't have a terminal parent
    #                 continue
    #             parents += [self.obs(sp)]
    #             actions += [i]

    #     return parents, actions

    def parent_transitions(self,s):
            parents = []
            actions = []
            results = np.where(s==1)
            
            if len(results[0])==0:
                parents.append(s)
                actions.append(-1)
                return s , actions

            for item in results[0]:
                sp = s+0
                sp[item] -= 1
                
                parents.append(sp)
                actions.append(item)

            return parents, actions

    def step(self, a, s = None):
        return self.step_dag(a, s)

    # def step_dag(self, a, s=None):
    #     _s = s
    #     s = (self._state if s is None else s) + 0
    #     if a < self.ndim:

    #         s[a] += 1

    #     done = s.max() >= self.horizon - 1 or a == self.ndim

    #     if _s is None:
    #         self._state = s
    #         self._step += 1

    #     return self.obs(s), 0 if not done else self.func(self.s2x(s)), done, s

    def step_dag(self,a, s= None):
        no_step = False
        print('s1',s)
        print('a1',a)
        _s = s
        s = (tf(np.zeros((self.n_node * self.n_node))) if s is None else s) + 0
        # s = (self._state if s is None else s) + 0
        # sample 没超过维度且需要没有选中重复的
        # if a < len(s) and s[int(a)]==0:
        print()
        if s[int(a)]==0:
            new_s = s+0
            new_s[int(a)] = 1

            # matrix_new_s = np.array(new_s).reshape(args.n_node,args.n_node)

            row_ix = int(a)//args.n_node
            # print('row_ix',row_ix)
            col_ix = int(a)% args.n_node # 3 is dim 
            # print('col_ix',col_ix)
            Trans_ix = col_ix * args.n_node + row_ix
            
            # print('Trans_ix',Trans_ix)
            # 看加了该元素时，是否让对角线为1, 看是否让对称的元素重复为1,如果是则说明结束了（不能走）
            #if len(np.where(np.diag(matrix_new_s)==1)[0]) >0 or new_s[Trans_ix] == 1:
            if new_s[Trans_ix] == 1:
                no_step = True
            
        else:
            # sample超过维度了
            no_step = True

        # if _s is None:
        #     self._state = s
        #     self._step += 1    
        
        return s if no_step else new_s, cal_reward_simple(np.array(s).reshape(args.n_node,args.n_node)) if no_step else 0 ,no_step

            
    
    # def true_density(self):
    #     """
    #     Computes the reward density (reward / sum(rewards)) of the whole space

    #     """
    #     if self._true_density is not None:
    #         return self._true_density
    #     all_int_states = np.int32(list(itertools.product(*[list(range(self.horizon))]*self.ndim)))
    #     state_mask = np.array([len(self.parent_transitions(s, False)[0]) > 0 or sum(s) == 0
    #                            for s in all_int_states])
    #     all_xs = (np.float32(all_int_states) / (self.horizon-1) *
    #               (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
    #     traj_rewards = self.func(all_xs)[state_mask]
    #     self._true_density = (traj_rewards / traj_rewards.sum(),
    #                           list(map(tuple,all_int_states[state_mask])),
    #                           traj_rewards)
    #     return self._true_density

    # def all_possible_states(self):
    #     """Compute quantities for debugging and analysis"""
    #     # all possible action sequences
    #     def step_fast(a, s):
    #         s = s + 0
    #         s[a] += 1
    #         return s
    #     f = lambda a, s: (
    #         [np.int32(a)] if np.max(s) == self.horizon - 1 else
    #         [np.int32(a+[self.ndim])]+sum([f(a+[i], step_fast(i, s)) for i in range(self.ndim)], []))
    #     all_act_seqs = f([], np.zeros(self.ndim, dtype='int32'))
    #     # all RL states / intermediary nodes
    #     all_int_states = list(itertools.product(*[list(range(self.horizon))]*self.ndim))
    #     # Now we need to know for each partial action sequence what
    #     # the corresponding states are. Here we can just count how
    #     # many times we moved in each dimension:
    #     all_traj_states = np.int32([np.bincount(i[:j], minlength=self.ndim+1)[:-1]
    #                                for i in all_act_seqs
    #                                for j in range(len(i))])
    #     # all_int_states is ordered, so we can map a trajectory to its
    #     # index via a sum
    #     arr_mult = np.int32([self.horizon**(self.ndim-i-1)
    #                          for i in range(self.ndim)])
    #     all_traj_states_idx = (
    #         all_traj_states * arr_mult[None, :]
    #     ).sum(1)
    #     # For each partial trajectory, we want the index of which trajectory it belongs to
    #     all_traj_idxs = [[j]*len(i) for j,i in enumerate(all_act_seqs)]
    #     # For each partial trajectory, we want the index of which state it leads to
    #     all_traj_s_idxs = [(np.bincount(i, minlength=self.ndim+1)[:-1] * arr_mult).sum()
    #                        for i in all_act_seqs]
    #     # Vectorized
    #     a = torch.cat(list(map(torch.LongTensor, all_act_seqs)))
    #     u = torch.LongTensor(all_traj_states_idx)
    #     v1 = torch.cat(list(map(torch.LongTensor, all_traj_idxs)))
    #     v2 = torch.LongTensor(all_traj_s_idxs)
    #     # With all this we can do an index_add, given
    #     # pi(all_int_states):
    #     def compute_all_probs(policy_for_all_states):
    #         """computes p(x) given pi(a|s) for all s"""
    #         dev = policy_for_all_states.device
    #         pi_a_s = torch.log(policy_for_all_states[u, a])
    #         q = torch.exp(torch.zeros(len(all_act_seqs), device=dev)
    #                                   .index_add_(0, v1, pi_a_s))
    #         q_sum = (torch.zeros((all_xs.shape[0],), device=dev)
    #                  .index_add_(0, v2, q))
    #         return q_sum[state_mask]
    #     # some states aren't actually reachable
    #     state_mask = np.bincount(all_traj_s_idxs, minlength=len(all_int_states)) > 0
    #     # Let's compute the reward as well
    #     all_xs = (np.float32(all_int_states) / (self.horizon-1) *
    #               (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
    #     traj_rewards = self.func(all_xs)[state_mask]
    #     # All the states as the agent sees them:
    #     all_int_obs = np.float32([self.obs(i) for i in all_int_states])
    #     print(all_int_obs.shape, a.shape, u.shape, v1.shape, v2.shape)
    #     return all_int_obs, traj_rewards, all_xs, compute_all_probs



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

# class SplitCategorical:
#     def __init__(self, n, logits):
#         """Two mutually exclusive categoricals, stored in logits[..., :n] and
#         logits[..., n:], that have probability 1/2 each."""
#         self.cats = Categorical(logits=logits[..., :n]), Categorical(logits=logits[..., n:])
#         self.n = n
#         self.logits = logits

#     def sample(self):
#         split = torch.rand(self.logits.shape[:-1]) < 0.5
#         return self.cats[0].sample() * split + (self.n + self.cats[1].sample()) * (~split)

#     def log_prob(self, a):
#         split = a < self.n
#         log_one_half = -0.693147
#         return (log_one_half + # We need to multiply the prob by 0.5, so add log(0.5) to logprob
#                 self.cats[0].log_prob(torch.minimum(a, torch.tensor(self.n-1))) * split +
#                 self.cats[1].log_prob(torch.maximum(a - self.n, torch.tensor(0))) * (~split))

#     def entropy(self):
#         return Categorical(probs=torch.cat([self.cats[0].probs, self.cats[1].probs],-1) * 0.5).entropy()

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
        print('s_before',s)
        done = [False] * mbsize
        while not all(done):
            print('done',done)
            # Note to self: this is ugly, ugly code
            with torch.no_grad():
                print('s_during',s)
                model_outcome = self.model(tf(s))
                print('model_outcome',model_outcome)
                acts = Categorical(logits=model_outcome).sample()
            # print('s2',s)
            print('acts',acts)
            # for i,a in zip([env for done, env in zip(done, self.envs) if not done], acts):
            #     step = i.step(a)
            
            step = [i.step(a) for i,a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
        
            print('step',step)
            print('len(step)',len(step))
            # sp1, r1, done1 = step[0]
            # print('sp1',sp1)
            # print('r1',r1)
            # print('done1',done1)
            p_a = [self.envs[0].parent_transitions(sp)
                   for a, (sp, r, done) in zip(acts, step)]
            print('p_a',p_a)
            # print('lenth(p_a)',len(p_a))


            # sp, r, d = step[0]

            # if ac[0]== -1:
            #     print('-1')
            #     break
            # print('p_a[0]',p_a[0])
            # for (p, a), (sp, r, d) in zip(p_a, step):

            #     print('p',p)
               
            #     print('tf(p)',tf(p))
            #     print('################')
            #     print('a',a)
            #     print('tf(a)',tf(a))
            #     print('################')
            #     print('r',[r])
            #     print('tf(r)',tf([r]))
            #     print('################')
            #     print('sp',sp)

            #     print('tf(sp)',tf(sp))
            #     print('################')
            #     print('d',d)
            #     print('tf(d)',tf([d]))

            batch += [[ i for i in (p, tf(a), tf([r]), sp, tf([d]))]
                      for (p, a), (sp, r, d) in zip(p_a, step)]
            print('batch',batch)
            # batch += [[tf(i) for i in (p, a, [r], [sp], [d])]
            #           for (p, a), (sp, r, d) in zip(p_a, step)]
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]

            for i in step:
                if not i[2]:
                    s = i [0]
            print('s_after',s)

            for (sp, r, d) in step:
                if d:
                    all_visited.append(tuple(sp))
                    self.replay.add(tuple(sp), r)

        return batch

    def learn_from(self, it, batch):
        loginf = torch.Tensor([1000])

        batch_idxs = tl(sum([[i]*len(parents) for i, (parents,_,_,_,_) in enumerate(batch)], []))
        # print('torch.cat',torch.cat)
        # print('map(torch.cat, zip(*batch))',map(torch.cat, zip(*batch)))
        #parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        # print('test_here')
        if actions[0] == -1:
            return None

        parents_Qsa = self.model(torch.tensor(parents))  [torch.arange(parents.shape[0]), actions.long()]
        
        # print('parents_Qsa',parents_Qsa)
        # a = actions.long()
        # test_value =torch.zeros((sp.shape[0],))
        in_flow = torch.log(torch.zeros((sp.shape[0],))
                            .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))

        next_q = self.model(torch.tensor(sp))

        next_qd = next_q * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)

        out_flow = torch.logsumexp(   torch.cat(  [torch.log( r)[:, None], next_qd  ], 1)  , 1)
        print('inflow',in_flow)
        print('outflow',out_flow)
        loss = (in_flow - out_flow).pow(2).mean() # explosion

        # print('test_3')
        # loss = abs((in_flow - out_flow)).mean()
        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((in_flow - out_flow) * (1-done)).pow(2).sum() / ((1-done).sum() + 1e-20)

        return loss, term_loss, flow_loss



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


# def compute_empirical_distribution_error(env, visited):
#     if not len(visited):
#         return 1, 100
#     hist = defaultdict(int)
#     for i in visited:
#         hist[i] += 1
#     td, end_states, true_r = env.true_density()
#     true_density = tf(td)
#     Z = sum([hist[i] for i in end_states])
#     estimated_density = tf([hist[i] / Z for i in end_states])
#     k1 = abs(estimated_density - true_density).mean().item()
#     # KL divergence
#     kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
#     return k1, kl

def main(args):
    weighted_random_dag = DAG.erdos_renyi(n_nodes=args.n_node,n_edges=12,weight_range=(0.5,2.0),seed=1)
    # weighted_random_dag
    datasets = IIDSimulation(weighted_random_dag,n=50,method='linear',sem_type = 'gauss')
    # datasets

    true_causal_matrix,X = datasets.B,datasets.X
    #true_causal_matrix

    # X = Tensor(X)
    n_samples = X.shape[0]
    seq_length = X.shape[1]

    # print('n_samples',n_samples)
    # print('seq_length',seq_length)

    data_generator = DataGenerator(dataset=X,normalize=False)
    data_generator
    input_batch = data_generator.draw_batch(batch_size=5,dimension=100)
    input_batch.shape
    args.dev = torch.device(args.device)
    set_device(args.dev)


    reward =Reward(input_data=data_generator.dataset.cpu().detach().numpy(),
                    reward_mode='dense',
                    score_type='BIC',
                    regression_type='LR',
                    alpha=1.0)

    env = CausalEnv(args.n_node)
    envs = [CausalEnv(args.n_node) for i in range(args.bufsize)]
    # env  = [gym.make('CartPole-v0') for i in range(args.bufsize)]

    if args.method == 'flownet':
        agent = FlowNetAgent(args, envs)

    opt = make_opt(agent.parameters(), args)

    # metrics
    all_losses = []
    all_visited = []
    # empirical_distrib_losses = []

    ttsr = max(int(args.train_to_sample_ratio), 1) # 1
    sttr = max(int(1/args.train_to_sample_ratio), 1) # sample to train ratio
    #
    # if args.method == 'ppo':
    #     ttsr = args.ppo_num_epochs
    #     sttr = args.ppo_epoch_size

    start_time = time.time()
    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        data = []
        for j in range(sttr):

            data += agent.sample_many(args.mbsize, all_visited) # mbsize = 16
        for j in range(ttsr):
            losses = agent.learn_from(i * ttsr + j, data) # returns (opt loss, *metrics)
            if losses is not None:
                print('losses[0]==',losses[0])
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
