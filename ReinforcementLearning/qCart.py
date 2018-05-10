import gym
import numpy as np
import math
from collections import deque

class QCartPoleSolver():

    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=195, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25, max_env_steps=None, quiet=False, monitor=False):
        self.buckets = buckets # down-scaling feature space to discrete range
        #buckets -> quantos valores são possiveis para cada feature da observação? Para as duas primeiras features , apenas 1. Para a terceira, 6 valores e para a quarta, 12.

        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.quiet = quiet

        self.env = gym.make('CartPole-v0')
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

        #tabela começa preenchida por zeros.
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    #retorna a observação do ambiente especificada discretizada (valores inteiros)
    def discretize(self, obs):


        #----------------------------Limitação do espaço de observações----------------------------
        
        #a posicao do carrinho não é limitada pela discretização
        #a velocidade é limitada superiormente por 0.5 e inferiormente por -0.5
        #o angulo não é limitado pela discretização
        #a velocidade angular é limitada superiormente por radians(50) e inferiormente por -0.5
        

        #upper bounds é uma lista composta pelo valor máximo que cada dimensao de uma observação pode assumir
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        '''
            ^
            |
        # as listas tem 4 elementos pois o ambiente tem 4 observações (posicao, velocidade, angulo e velocidade angular)
            |
            v
        '''
        #lower bounds é uma lista composta pelo valor mínimo que cada dimensao de uma observação pode assumir
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]

        #--------------------------------------------------------------------------------------------

        #ratios é uma lista contendo as proporções da observação a ser discretizada com relação ao tamanho de cada dimensão.
        #suponha que a observação a ser discretizada seja (2, 5, 0.3, 8)

        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]

        ''' os ratios dessa observação seriam ( ( (2 + 4.8) / (4.8 + 4.8) ),
                                                ( (5 + 0.5) / (0.5 + 0.5) ),
                                                ( (0.3 + 0.42) / (0.42 + 0.42) ),
                                                ( (8 + 0.87) / (0.87 + 0.87) )

        = (0.7, 5.5, 0.8, 5.09)
        '''

        #a discretização acontece justamente no processo de arredondamento.
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        #new_obs = (0, 0, 5*0.8, 11 * 5.09  ) = (0, 0, 4, 56)
        
        

        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        '''new_obs = [min (0, max(0, 0) ),
                      min (0, max(0, 0) ),
                      min (5, max(0, 4) ),
                      min (11, max(0, 56)]

            = [0, 0, 4, 11]
        
        '''
        


        return tuple(new_obs)

    #a acao a ser escolhida será  ou uma ação aleatoria ou a de maior q-value
    def choose_action(self, state, epsilon):
        #Q[state] -> retorna um array contendo os q-values de cada açao possível a partir do estado state. -> Duas ações possíveis, mover para a esquerda 0 ou para a direita 1
        #np.argmax(array) -> retorna o índice do maior elemento do array -> array de duas posições, pode retornar 0 ou 1
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])


    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):
        scores = deque(maxlen=100)

        #em cada episode, ou o problema é resolvido, ou o controle falha em manter o pole balanceado
        for e in range(self.n_episodes):

            #o estado atual é uma discretização do estado retornado por env.reset()
            #env.reset retorna uma observação inicial do ambiente
            current_state = self.discretize(self.env.reset())

            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            while not done:
                self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials '.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        if not self.quiet: print('Did not solve after {} episodes -- sad'.format(e))
        return e

if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run()