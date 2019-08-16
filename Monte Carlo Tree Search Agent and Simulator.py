#import necessary modules
from copy import copy, deepcopy
import datetime
import math
import random
import numpy as np
import sys


#We create a field class in our case it incorporates the rules to calulate the yield and the possible next actions that can be taken. It acts as a simulator.

class Field(object):
    def __init__(self, n):
        # Initializes an empty field of size n*n with each entry as '-'
        '''
        input n : size of the field
        '''
        self.n = n
        self.matrix = np.matrix(np.full([n,n],'-'))
            
    def next_state(self, state, move):
        # Takes the current state of the field, the crop to be planted next and its location
        # Returns the new field state.
        '''
        input state : indicates the current state of the field
        input move : indicates next move i.e to palnt either corn or bean at i and j indices
        output : a new state of the field 
        '''
        next_state = deepcopy(state)
        next_state.matrix[move[1],move[2]] = move[0]
        return next_state

    def possible_next_actions(self, state_history):
        # Takes the history of field states and
        # Returns all the feasible actions that can be taken to get next state
        '''
        input state_history : a list of states indicating the hsitory of the field
        output : a list of actions like planting bean or corn at i and j indices of the field 
        '''
        state = state_history[-1]
        legal_actions = []
        for i in range(self.n):
            for j in range(self.n):
                if state.matrix.item(i,j) == '-':
                    legal_actions.append(('b',i,j))
                    legal_actions.append(('c',i,j))
        return legal_actions
    
    def possible_next_states(self, state_history):
        # Takes a sequence of field states representing the full
        # field history, and returns the full list of next states that
        # are feasible
        '''
        input state_history : a list of list indicating the current state of the field
        output : a list of all next possible field states 
        '''
        possible_states = []
        legal_actions = self.possible_next_actions(state_history)
        state = state_history[-1][:]
        for action in legal_actions:
            state.matrix[action[1],action[2]] = action[0]
            possible_states.append(state)
            state = state_history[-1][:]
        return possible_states
        

    def is_full(self, state):
        # Takes current state of the field 
        # and returns if the field is full or not
        '''
        input state : indicates current state of field
        output : true if field is full else returns false 
        '''
        for i in range(self.n):
            for j in range(self.n):
                if state.matrix.item(i,j) == '-':
                    return False
        return True
    
    def calculate_total_yield(self,state):
        """
        input state : indicates the current state of the field
        output total_yield : total yield of crops in the field 
        """
        M = state.matrix
        total_yield = 0
        for i in range(self.n):
            for j in range(self.n):
                # Case 1 : if there is a corn crop in the i,j cell of the field
                if M.item(i,j) == 'c':
                    # Following code checks the number of bean crops adjacent to the corn crop 
                    adjacent_beans = 0
                    k = i-1
                    l = j-1
                    while k <= i+1:
                        if k >= 0 and k < self.n :
                            while l <= j+1:
                                if l >= 0 and l < self.n:
                                    if M.item(k,l) == 'b':
                                        adjacent_beans += 1
                                l += 1
                        l = j-1
                        k += 1
                    total_yield += 10 + adjacent_beans

                # Case 2 : if there is a bean crop in the i,j cell of the field
                if M.item(i,j) == 'b':
                    # Following code checks the number of corn crops adjacent to the bean crop 
                    adjacent_corns = 0
                    k = i-1
                    l = j-1
                    while k <= i+1:
                        if k >= 0 and k < self.n:
                            while l <= j+1:
                                if l >= 0 and l < self.n:
                                    if M.item(k,l) == 'c':
                                        adjacent_corns += 1
                                l += 1
                        l = j-1
                        k += 1
                    if adjacent_corns > 0:
                        total_yield += 15
                    else:
                        total_yield += 10
                #print (i,j,M[i][j],total_yield)
        return total_yield
		
		
#We create a Monte Carlo class which incorporates the logic of a Monte Carlo Tree Search Agent and it does not how the simulator works. It queries the simulator to know the possible next actions and their rewards. Based on the reward values it takes decisions to generate a field plan

class MonteCarlo(object):
    def __init__(self, field, **kwargs):
        # Takes an instance of a Field and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics trees.
        self.field = field
        self.states = [field]
        seconds = kwargs.get('time', 30)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 150)
        self.simulation_statistics = {tuple(field.matrix.A1):[0,1]}
        
    def update(self, state):
        # Takes a game state and appends it to the state history.
        self.states.append(state)

    def grow_plant(self):
        # Causes the AI to calculate the best move from the
        # current game state and return it
        # it is implementation of UTC algorithm which relies on playing multiple games from the current state 
        self.max_depth = 0
        state = deepcopy(self.states[-1])
        actions = self.field.possible_next_actions(self.states[:])
        
        # return early if there is no choice or there is only one choice
        if not actions:
            return 
        if len(actions) == 1:
            return actions[0]
            
        begin = datetime.datetime.utcnow()
        total_simulation = 0
        
        # run simulations until we run out of time
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation()
            total_simulation += 1
        
        moves_states = [(p,self.field.next_state(state,p)) for p in actions]
        #print([(self.simulation_statistics[tuple(S.matrix.A1)][0],p) for p,S in moves_states])
        max_yeild = -1
        move = None
        for p,S in moves_states:
            if max_yeild < self.simulation_statistics[tuple(S.matrix.A1)][0]:
                max_yeild = self.simulation_statistics[tuple(S.matrix.A1)][0]
                move = p
        return move

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        states_copy = deepcopy(self.states)
        state = deepcopy(states_copy[-1])
        
        # expand tree till depth of max_moves
        for t in range(self.max_moves):
            # selection step to select a node randomly or based on UCB value
            actions = self.field.possible_next_actions(states_copy)
            next_states = [(p, self.field.next_state(state, p)) for p in actions]
            action = None
            
            if set([tuple(next_state[1].matrix.A1) for next_state in next_states]).issubset(set(self.simulation_statistics.keys())):
                # if we have stats on all of the next actions we use them to calculate average of current node and select best node based on UCB1 formula
                total_reward = sum(self.simulation_statistics[tuple(next_state.matrix.A1)][0]*self.simulation_statistics[tuple(next_state.matrix.A1)][1] for _,next_state in next_states)
                total_simulations = sum(self.simulation_statistics[tuple(next_state.matrix.A1)][1] for _,next_state in next_states)
                
                # updating the statistics of parent node based on all the values of the children nodes
                self.simulation_statistics[tuple(state.matrix.A1)] = [total_reward,total_simulations]
                
                # finding a node with highest UCB value as next state
                max_UCB1_value = -1
                for p,s in next_states:
                    state_UCB1_value = (math.log(self.simulation_statistics[tuple(state.matrix.A1)][1])/self.simulation_statistics[tuple(s.matrix.A1)][1] + self.simulation_statistics[tuple(s.matrix.A1)][1])
                    if max_UCB1_value < state_UCB1_value:
                        max_UCB1_value = state_UCB1_value
                        action = p
            else:
                # else we make an arbitrary decision and select a node which has not yet been explored
                unexplored_actions = []
                for action in actions:
                    if tuple(self.field.next_state(state, action).matrix.A1) not in self.simulation_statistics.keys():
                        unexplored_actions.append(action)
                action = unexplored_actions[random.randint(0,len(unexplored_actions)-1)]
            
            # finding the next state and appending it to states_copy
            state = self.field.next_state(state, action)
            states_copy.append(state)
            
            # stop if the field is full and there are no more moves
            if self.field.is_full(states_copy[-1]) or actions == None:
                break
        
        # code for back propogation to update the statistics tree for all nodes that are visited in last iteration of simulation
        # we will update statistically value of all the nodes from leaf to root on this path
        # the last node of the states_copy will be a terminal node because that is the only reason we stopped the simulation and no next action is possible
        # so we directly add it to the dictionary and if it is already in the dictionary we just increment it's number of simulations
        if tuple(states_copy[-1].matrix.A1) not in self.simulation_statistics.keys():
            self.simulation_statistics[tuple(state.matrix.A1)] = [self.field.calculate_total_yield(state),1]
        elif tuple(states_copy[-1].matrix.A1) in self.simulation_statistics.keys():
            self.simulation_statistics[tuple(state.matrix.A1)][1] += 1

        for index in reversed(range(len(states_copy)-1)):
            state = states_copy[index]
            next_state = states_copy[index + 1]
            # if the current state is visited for the first time during expansion we add its value to the statistical tree
            if tuple(state.matrix.A1) not in self.simulation_statistics.keys():
                self.simulation_statistics[tuple(state.matrix.A1)] = [self.simulation_statistics[tuple(next_state.matrix.A1)][0],self.simulation_statistics[tuple(next_state.matrix.A1)][1]]
            # otherwise we update its statistics in the dictionary
            # in order to update the average yeild of the leaf node we take average yeild of all its children node
            else:
                key = tuple(state.matrix.A1)
                original_total = self.simulation_statistics[key][1] * self.simulation_statistics[key][0]
                new_total = original_total + self.simulation_statistics[tuple(next_state.matrix.A1)][0]
                self.simulation_statistics[key][0] = (new_total) / (self.simulation_statistics[key][1] + 1)
                self.simulation_statistics[key][1] += 1
                

n =  int(sys.argv[1])				
F = Field(n)
AI_Agent = MonteCarlo(F, time = 30, max_moves = n*n)


while not AI_Agent.states[-1].is_full(AI_Agent.states[-1]):
    action = AI_Agent.grow_plant()
    print("Next Optimum move was: ",AI_Agent.states[-1].next_state(AI_Agent.states[-1],action).matrix)
    AI_Agent.update(AI_Agent.states[-1].next_state(AI_Agent.states[-1],action))
    
	
print("states visited are as followed")
for state in AI_Agent.states:
    print(state.matrix)