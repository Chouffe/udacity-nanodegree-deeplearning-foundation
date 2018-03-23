import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, gamma=1.0, alpha=0.1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """

        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = gamma
        self.alpha = alpha
        self.next_action = None


    def __update_rule_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        """
        Updates the action-value function estimate using the most recent time step
        """

        return Qsa + alpha * (reward + (gamma * Qsa_next) - Qsa)


    def __epsilon_greedy_probs(self, nA, Q_s, i_episode=1):
        """
        Obtains the action probabilities corresponding to epsilon-greedy policy
        """

        epsilon = 1.0 / i_episode
        policy_s = np.ones(nA) * epsilon / nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / nA)

        return policy_s


    def select_action(self, state, i_episode=1):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if self.next_action:
            return self.next_action
        else:
            policy_s = self.__epsilon_greedy_probs(self.nA, self.Q[state], i_episode=i_episode)
            action = np.random.choice(np.arange(self.nA), p=policy_s)

            return action


    def sarsa_step(self, state, action, reward, next_state, done, i_episode=1):
        """
        Update the agent's knowledge, using the most recently sampled tuple using SARSA

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if not done:

            # Get epsilon-greedy action probabilities
            policy_s = self.__epsilon_greedy_probs(self.nA, self.Q[next_state], i_episode=i_episode)
            next_action = np.random.choice(np.arange(self.nA), p=policy_s)

            # Update TD estimate of Q
            self.Q[state][action] = self.__update_rule_Q(
                self.Q[state][action],
                self.Q[next_state][next_action],
                reward,
                self.alpha,
                self.gamma
            )

            # Add next action to use for the agent
            self.next_action = next_action

        else:
            self.Q[state][action] = self.__update_rule_Q(
                self.Q[state][action],
                0,
                reward,
                self.alpha,
                self.gamma
            )

            # Reset Next action
            self.next_action = None


    def sarsamax_step(self, state, action, reward, next_state, done, i_episode=1):
        """
        Update the agent's knowledge, using the most recently sampled tuple using SARSAMAX - QLearning

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:

            # Get epsilon-greedy action probabilities
            policy_s = self.__epsilon_greedy_probs(self.nA, self.Q[next_state], i_episode=i_episode)
            next_action = np.random.choice(np.arange(self.nA), p=policy_s)

            # Update TD estimate of Q
            self.Q[state][action] = self.__update_rule_Q(
                self.Q[state][action],
                np.max(self.Q[next_state]),
                reward,
                self.alpha,
                self.gamma
            )

            # Add next action to use for the agent
            self.next_action = next_action

        else:
            self.Q[state][action] = self.__update_rule_Q(
                self.Q[state][action],
                0,
                reward,
                self.alpha,
                self.gamma
            )

            # Reset Next action
            self.next_action = None


    def expected_sarsa_step(self, state, action, reward, next_state, done, i_episode=1):
        """
        Update the agent's knowledge, using the most recently sampled tuple using SARSAMAX - QLearning

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if not done:

            # Get epsilon-greedy action probabilities
            policy_s = self.__epsilon_greedy_probs(self.nA, self.Q[next_state], i_episode=i_episode)
            next_action = np.random.choice(np.arange(self.nA), p=policy_s)

            # Update TD estimate of Q
            self.Q[state][action] = self.__update_rule_Q(
                self.Q[state][action],
                np.dot(self.Q[next_state], policy_s),
                reward,
                self.alpha,
                self.gamma
            )

            # Add next action to use for the agent
            self.next_action = next_action

        else:
            self.Q[state][action] = self.__update_rule_Q(
                self.Q[state][action],
                0,
                reward,
                self.alpha,
                self.gamma
            )

            # Reset Next action
            self.next_action = None


    def step(self, state, action, reward, next_state, done, i_episode=1):
        """
        Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # Use one of the following update steps by uncommenting it
        # self.sarsa_step(state, action, reward, next_state, done, i_episode=i_episode)
        # self.sarsamax_step(state, action, reward, next_state, done, i_episode=i_episode)
        self.expected_sarsa_step(state, action, reward, next_state, done, i_episode=i_episode)
