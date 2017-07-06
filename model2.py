'''
model2.py, v1.0.1, 17/07/06, by Max Murakami
    written in Python 2.7.12

Agent class for simulating action selection with intrinsic motivation based on
    information maximization. The model is based on Bolado-Gomez & Gurney 2013
    with these modifications:
        - novelty salience is defined as expected information gain given the
            current predictions (was originally piecewise linear function of 
            predictions),
        - the neural network component is removed along with the sensory prediction
            error signal,
        - exploration uses softmax instead of uniform probability distribution.

Usage:
    - Specify contingencies in environment_response() method.
    - During construction, specify either initial intrinsic action saliences
        (initial_int_sal), initial prediction variables (initial_probs), or both.
        Both initial_int_sal and initial_probs must be 1 dimensional float ndarrays.
        Their length determines the number of actions (lengths must match if both
        are specified).
    - Free parameters:
        - exploration_rate: the higher, the more likely the agent executes actions
            with low saliences (softmax exploration)
        - learning_speed: the higher, the faster the agent acquires the contingencies
            -> time constant of novelty salience
        - hab_slowness: the higher, the less intrinsic saliences decrease due to habituation
            -> time constant of intrinsic salience
    - These command line arguments are available if you run this script from the terminal:
        - v: turns on verbose mode
        - o: turns on file output mode
        - T [int]: specify the number of simulation time steps

Other files:
    - aux.py contains auxillary functions.
    - unit_test.py contains unit tests. Run these when modifying the code.
    - output files are created in ./output/[date]/[time]/ folder:
        - output_constants.txt contains constant values of object
        - output.txt contains variables of object over time
        - output.dat contains pickled dictionary with variables and constants

Version history:
    - 1.0.1:
        - added reinit() method
        - record() and run() now always return data dict, independent of --output
'''

import numpy as np
import aux      # auxillary functions
import cPickle


class Agent:
    def __init__(self, initial_int_sal=None, initial_probs=None, exploration_rate=1.,
        learning_speed=0.9, hab_slowness=0.9, T_max=int(1e4), verbose=False, output=False):
        '''
        constructor

        initial_int_sal is initial distribution of intrinsic saliences for all actions
            type is list or np.ndarray, default is np.ones([4])
        initial_probs is initial values of contingency predictions for all actions
            type is list or np.ndarray, length must match initial_int_sal, default is np.zeros([4])
        exploration_rate is used for softmax action selection
            type is float, default is 1
        learning_speed is adaptation rate of prediction system (float), default is 0.9
        hab_slowness is factor for action habituation (float), default is 0.9,
            high means slow habituation, low means fast habituation
        T_max is number of time steps (int), default is 1e4
        verbose is verbose output mode
            type is boolean, default is True
        output is file output mode
            type is boolean, default is True
        '''

        if verbose:
            print 'Constructing Agent object. Received arguments:'
            if isinstance(initial_int_sal, np.ndarray) or initial_int_sal:
                print ' initial_int_sal:', initial_int_sal
            if isinstance(initial_probs, np.ndarray) or initial_probs:
                print ' initial_probs:', initial_probs
            if isinstance(exploration_rate, np.ndarray) and exploration_rate!=1.:
                print ' exploration_rate:', exploration_rate
            if isinstance(learning_speed, np.ndarray) and learning_speed!=0.9:
                print ' learning_speed:', learning_speed
            if isinstance(hab_slowness, np.ndarray) and hab_slowness!=0.9:
                print ' hab_slowness:', hab_slowness
            if isinstance(T_max, np.ndarray) or T_max != 1e4:
                print ' T_max:', T_max


            ####### initial_int_sal
        if aux.isNotFalse(initial_int_sal):     # user specified initial_int_sal
            assert isinstance(initial_int_sal, list) or isinstance(initial_int_sal, np.ndarray),\
                "\nInput 'initial_int_sal' ({}) of Agent constructor has wrong type ({})!\
                 \nType must be 'list' or 'np.ndarray'!".format(initial_int_sal, type(initial_int_sal))

            if isinstance(initial_int_sal, list):
                initial_int_sal = np.array(initial_int_sal)

            aux.check_that_1d_float_array(initial_int_sal, 'initial_int_sal')

            self.initial_int_sal = np.array(initial_int_sal, dtype=float)

        elif initial_int_sal == None:       # user didn't specify initial_probs
            if aux.isNotFalse(initial_probs):
                assert isinstance(initial_probs, list) or isinstance(initial_probs, np.ndarray),\
                    "\nInput 'initial_probs' ({}) of Agent constructor has wrong type ({})!\
                    \nType must be 'list' or 'np.ndarray'!".format(initial_probs, type(initial_probs))
                self.initial_int_sal = np.ones([len(initial_probs)])
            else:
                self.initial_int_sal = np.ones([4])
        else:
            raise AssertionError,\
                "\nInput 'initial_int_sal' ({}) of Agent constructor has wrong type ({})!\
                 \nType must be 'list' or 'np.ndarray'!".format(initial_int_sal, type(initial_int_sal))

        self.int_sal = self.initial_int_sal     # initial saliences that are updated
        self.N_actions = len(self.initial_int_sal)


            ####### initial_probs
        if aux.isNotFalse(initial_probs):   # user specified initial_probs
            assert isinstance(initial_probs, list) or isinstance(initial_probs, np.ndarray),\
                "\nInput 'initial_probs' ({}) of Agent constructor has wrong type ({})!\
                \nType must be 'list' or 'np.ndarray'!".format(initial_probs, type(initial_probs))

            if isinstance(initial_probs, list):
                initial_probs = np.array(initial_probs)

            aux.check_that_1d_float_array(initial_probs, 'initial_probs')
            aux.check_that_contains_probabilities(initial_probs, 'initial_probs')

            self.initial_probs = np.array(initial_probs, dtype=float)

            assert len(initial_probs) == self.N_actions,\
                "\nInput 'initial_probs' ({}) of Agent constructor has wrong length ({})!\
                \nLenth must match length of 'initial_int_sal' ({})!".format(initial_probs,
                    len(initial_probs), self.N_actions)
                
        elif initial_probs == None:   # user didn't specify initial_probs
            self.initial_probs = np.zeros([self.N_actions])

        else:
            raise AssertionError,\
                "\nInput 'initial_probs' ({}) of Agent constructor has wrong type ({})!\
                \nType must be 'list' or 'np.ndarray'!".format(initial_probs, type(initial_probs))   

        self.probs = self.initial_probs     # probs that are updated
        self.nov_sal = self.get_novelty_saliences(self.probs)



            ####### exploration_rate
        assert aux.isNotFalse(exploration_rate),\
            "\nInput 'exploration_rate' of Agent constructor is False/None!"
        assert isinstance(exploration_rate, float) or isinstance(exploration_rate, int),\
            "\nInput 'exploration_rate' ({}) of Agent constructor has wrong type ({})!"\
            "\nType must be 'float'!".format(exploration_rate, type(exploration_rate))
        
        if isinstance(exploration_rate, int):
            exploration_rate = float(exploration_rate)

        assert exploration_rate > 0.0,\
            "\nInput 'exploration_rate' ({}) of Agent constructor must be greater "\
            "than 0!".format(exploration_rate)

        self.exploration_rate = exploration_rate



            ####### learning_speed
        assert aux.isNotFalse(learning_speed),\
            "\nInput 'learning_speed' of Agent constructor is False/None!"
        assert isinstance(learning_speed, float),\
            "\nInput 'learning_speed' ({}) of Agent constructor has wrong type ({})!"\
            "\nType must be 'float'!".format(learning_speed, type(learning_speed))
        assert learning_speed > 0.0 and learning_speed < 1.0,\
            "\nInput 'learning_speed' ({}) of Agent constructor is out of bounds!"\
            "\nlearning_speed must be in ]0;1[!".format(learning_speed)

        self.learning_speed = learning_speed



            ####### hab_slowness
        assert aux.isNotFalse(hab_slowness),\
            "\nInput 'hab_slowness' of Agent constructor is False/None!"
        assert isinstance(hab_slowness, float),\
            "\nInput 'hab_slowness' ({}) of Agent constructor has wrong type ({})!"\
            "\nType must be 'float'!".format(hab_slowness, type(hab_slowness))
        assert hab_slowness > 0.0 and hab_slowness < 1.0,\
            "\nInput 'hab_slowness' ({}) of Agent constructor is out of bounds!"\
            "\nhab_slowness must be in ]0;1[!".format(hab_slowness)

        self.hab_slowness = hab_slowness



            ####### T_max
        assert isinstance(T_max, int),\
            "\nInput 'T_max' ({}) of Agent constructor has wrong type ({})!"\
            "\nType must be 'int'!".format(T_max, type(T_max))
        assert T_max > 0,\
            "\nInput 'T_max' ({}) of Agent constructor is too small!"\
            "\nT_max must be positive!".format(T_max)

        self.t = 0
        self.T_max = T_max



            ####### output
        assert isinstance(output, bool),\
            "\nInput 'output' ({}) of Agent constructor has wrong type ({})!"\
            "\nType must be 'bool'!".format(output, type(output))
        self.output = output
        outputpath, time_string = aux.get_outputpath()

        # dictionary for storing all data
        self.data = {'exploration_rate':self.exploration_rate,
            'learning_speed':self.learning_speed, 'hab_slowness':self.hab_slowness}
        for label in ['int_sal', 'nov_sal', 'tot_sal', 'probs']:
            self.data.update({label:np.zeros([self.N_actions, self.T_max])})
        for label in ['i_select', 'env_response']:
            self.data.update({label:np.zeros([self.T_max], dtype=int)})

        if output:
            # output files are created in ./output/[date]/[time]/
            self.outputfile_txt = open('{}output.txt'.format(outputpath), 'w')
            outputfile_txt_const = open('{}output_constants.txt'.format(outputpath), 'w')
            self.outputfilename_dat = '{}output.dat'.format(outputpath)
            outputfile_dat = open(self.outputfilename_dat, 'wb')

            cPickle.dump(self.data, outputfile_dat)
            outputfile_dat.close()

                # create header of text file
            self.outputfile_txt.write('Time step\tSelected action\tEnvironment response')
            for name in ['Intrinsic saliences', 'Novelty saliences', 'Total saliences', 'Predictions']:
                self.outputfile_txt.write('\t{}'.format(name))
                for i in xrange(1,self.N_actions):
                    self.outputfile_txt.write('\t')
            self.outputfile_txt.flush()

                # write constants
            outputfile_txt_const.write('Start time:\t{}\nExploration rate:\t{}\nLearning speed:\t{}'\
                '\nHabituation slowness:\t{}'.format(time_string, self.exploration_rate,
                self.learning_speed, self.hab_slowness))
            outputfile_txt_const.close()




        self.random_numbers = np.random.rand(T_max)
        self.tot_sal = self.int_sal + self.nov_sal

        self.verbose = verbose


        if verbose:
            print 'Agent object constructed. Attributes are:'
            print 'initial_int_sal:', self.initial_int_sal
            print 'N_actions:', self.N_actions
            print 'initial_probs:', self.initial_probs
            print 'nov_sal:', self.nov_sal
            print 'tot_sal:', self.tot_sal
            print 'exploration_rate:', self.exploration_rate
            print 'learning_speed:', self.learning_speed
            print 'hab_slowness:', self.hab_slowness
            print 'T_max:', self.T_max
            print 'output:', self.output
            print 'Start time is', time_string




    def reinit(self, initial_int_sal=None, initial_probs=None):
        self.__init__(self, initial_int_sal=initial_int_sal, initial_probs=initial_probs,
            exploration_rate=self.exploration_rate, learning_speed=self.learning_speed, 
            hab_slowness=self.hab_slowness, T_max=self.T_max, verbose=self.verbose, 
            output=self.output)


    

    def select_action(self, tot_sal):
        '''
        softmax action selection

        tot_sal is ndarray of combined intrinsic and novelty saliences (ndarray of floats)

        returns i_selected_action, which is index of selected action (int)
        '''

        aux.check_that_1d_float_array(tot_sal, 'tot_sal')

        assert len(tot_sal)==self.N_actions,\
            "\nLength of salience vector input to select_action() ({}) does not match "\
            "number of actions ({})!".format(len(tot_sal), self.N_actions)


        ###### actual softmax
        action_probabilities = np.exp(tot_sal/self.exploration_rate)
        action_probabilities /= action_probabilities.sum()
            # normalize entries

        aux.check_that_contains_probabilities(action_probabilities, 'action_probabilities')
        assert abs(action_probabilities.sum() - 1.0) < 1e-6,\
            "\nAction probabilities don't sum up to 1: {}!"\
            "\nSum is {}!".format(action_probabilities, action_probabilities.sum())

        i_selected_action = np.random.choice(range(self.N_actions), p=action_probabilities)
            # samples an action given the probability distribution

        return i_selected_action



    def environment_response(self, i_action):
        '''
        determine environmental response to Agent's chosen action

        i_action is index of selected action (int)

        return value is either response (1) or no response (0)
        '''
        assert aux.isNotFalse(i_action),\
            "\nInput 'i_action' is False! i_action must be an int!"
        assert isinstance(i_action, int),\
            "\nInput 'i_action' ({}) of environment_response() has wrong type ({})!"\
            "\nType must be 'int'!".format(i_action, type(i_action))
        assert i_action in range(self.N_actions),\
            "\nInput 'i_action' ({}) of environment_response() has wrong value!"\
            "\ni_action must be in range({})!".format(i_action, self.N_actions)

        if i_action == 0:
            return 1
        elif i_action == 1:
            if self.random_numbers[self.t] > 0.5:
                return 1
            else:
                return 0
        else:
            return 0



    def update_predictions(self, i_action, env_response, probs):
        '''
        update prediction variables based on selected action and environment response

        i_action is index of selected action (int)
        env_response is environment response to action (binary int)
        probs is current prediction variables (ndarray of floats)

        returns: new_probs, updated prediction variables (ndarray of floats)
        '''
        assert aux.isNotFalse(i_action),\
            "\nInput 'i_action' to update_predictions() is False/None!"
        assert isinstance(i_action, int),\
            "\nInput 'i_action' ({}) to update_predictions() has wrong type ({})!"\
            "\nType must be 'int'!".format(i_action, type(i_action))
        assert i_action in range(self.N_actions),\
            "\nInput 'i_action' ({}) to update_predictions() is out of bounds!"\
            "\ni_action must be in range({})!".format(i_action, self.N_actions)
        assert aux.isNotFalse(env_response),\
            "\nInput 'env_response' to update_predictions() is False/None!"
        assert isinstance(env_response, int),\
            "\nInput 'env_response' ({}) to update_predictions() has wrong type ({})!"\
            "\nType must be 'int'!".format(env_response, type(env_response))
        assert env_response in range(2),\
            "\nInput 'env_response' ({}) to update_predictions() is out of bounds!"\
            "\nenv_reponse must be 0 or 1!".format(env_response)
        aux.check_that_1d_float_array(probs, 'probs')
        aux.check_that_contains_probabilities(probs, 'probs')
        assert len(probs) == self.N_actions,\
            "\nInput 'probs' ({}) to update_predictions() doesn't match number of actions!"\
            "\nLength is {} and must be {}!".format(probs, len(probs), self.N_actions)

        new_probs = probs.copy()
            # exponential prediction adaptation
        new_probs[i_action] = env_response - self.learning_speed * (env_response - probs[i_action])

        return new_probs
        



    def get_novelty_saliences(self, probs):
        '''
        calculate novelty saliences from contingency predictions

        probs is ndarray of contingency predictions, i.e. probabilities (ndarray of floats)

        returns: nov_saliences, contains novelty saliences, 
            i.e. expected information per action (ndarray of floats)
        '''

        aux.check_that_1d_float_array(probs, 'probs')
        aux.check_that_contains_probabilities(probs, 'probs')
        assert len(probs) == self.N_actions,\
            "\nInput 'probs' ({}) of get_novelty_saliences() doesn't match number of actions!"\
            "\nLength is {} and must be {}!".format(probs, len(probs), self.N_actions)

        nov_saliences = np.zeros([self.N_actions])
        for i in xrange(self.N_actions):
            p_i = probs[i]
            if p_i > 0.0 and p_i < 1.0:     # p_i=0 and p_i=1 cause unstable log computations,
                                            # nov_salience is 0 anyway in those cases
                nov_saliences[i] = 0.5 * (-p_i * np.log2(p_i) - (1.0-p_i) * np.log2(1.0-p_i))
                    # scaled binary entropy
        
        return nov_saliences




    def habituation(self, i_select, int_sal):
        '''
        update intrinsic saliences based on performed action

        i_select is index of selected action (int)
        int_sal is previous intrinsic saliences (ndarray of floats)

        returns: int_sal_new, updated intrinsic saliences (ndarray of floats)
        '''
        assert aux.isNotFalse(i_select),\
            "\nInput 'i_select' to habituation() is False/None!"
        assert isinstance(i_select, int),\
            "\nInput 'i_select' ({}) to habituation() has wrong type ({})!"\
            "\nType must be 'int'!".format(i_select, type(i_select))
        assert i_select in range(self.N_actions),\
            "\nInput 'i_select' ({}) to habituation() is out of bounds!"\
            "\ni_select must be in range({})!".format(i_select, self.N_actions)
        aux.check_that_1d_float_array(int_sal, 'int_sal')
        assert len(int_sal) == self.N_actions,\
            "\nInput 'int_sal' ({}) to habituation() doesn't match number of actions!"\
            "\nLength is {} and must be {}!".format(int_sal, len(int_sal), self.N_actions)
        assert (int_sal >= 0.0).all(),\
            "\nInput 'int_sal' ({}) to habituation() contains negative entries!".format(int_sal)
        assert (int_sal <= self.initial_int_sal).all(),\
            "\nInput 'int_sal' ({}) to habituation() is larger than "\
            "'initial_int_sal' ({})!".format(int_sal, self.initial_int_sal)

        int_sal_new = int_sal.copy()
        int_sal_new[i_select] = int_sal[i_select] * self.hab_slowness

        return int_sal_new




    def record(self):
        '''
        store variables in output files

        returns: data, dictionary containing Agent data
        '''
        
        # output dictionary
        data = self.data.copy()
        data['i_select'][self.t] = self.i_select
        data['env_response'][self.t] = self.response
        if self.verbose:
            print "recording:\n\tdata['i_select'][{}] = {}".format(self.t, self.i_select)
            print "\tdata['env_response'][{}] = {}".format(self.t, self.response)
        for i_action in xrange(self.N_actions):
            data['int_sal'][i_action][self.t] = self.int_sal[i_action]
            data['nov_sal'][i_action][self.t] = self.nov_sal[i_action]
            data['tot_sal'][i_action][self.t] = self.tot_sal[i_action]
            data['probs'][i_action][self.t] = self.probs[i_action]
            if self.verbose:
                print "\tdata['int_sal'][{}][{}] = {}".format(i_action, self.t, self.int_sal[i_action])
                print "\tdata['nov_sal'][{}][{}] = {}".format(i_action, self.t, self.nov_sal[i_action])
                print "\tdata['tot_sal'][{}][{}] = {}".format(i_action, self.t, self.tot_sal[i_action])
                print "\tdata['probs'][{}][{}] = {}".format(i_action, self.t, self.probs[i_action])

        if self.output:
            outputfile_dat = open(self.outputfilename_dat, 'wb')
            cPickle.dump(self.data, outputfile_dat)
            outputfile_dat.close()

            # output text file
            self.outputfile_txt.write('\n{}\t{}\t{}'.format(self.t, self.i_select, self.response))
            for item in [self.int_sal, self.nov_sal, self.tot_sal, self.probs]:
                for i_action in xrange(self.N_actions):
                    self.outputfile_txt.write('\t{}'.format(item[i_action]))
            self.outputfile_txt.flush()

        return data




    def run(self):
        '''
        perform action selection for T_max time steps

        returns: self.data, dictionary containing Agent data
        '''
        for t in xrange(self.T_max):
            self.t = t
            if self.verbose:
                print 'Time step:', self.t

            # action selection
            self.i_select = self.select_action(self.tot_sal)
            if self.verbose:
                print 'Agent performing action', self.i_select
        
            # response from environment
            self.response = self.environment_response(self.i_select)
            if self.verbose:
                print 'Environment response:', self.response

            # updating prediction variables
            self.probs = self.update_predictions(self.i_select, self.response, self.probs) 
            if self.verbose:
                print 'Updated predictions:', self.probs

            # computing novelty saliences
            self.nov_sal = self.get_novelty_saliences(self.probs)
            if self.verbose:
                print 'Novelty saliences:', self.nov_sal

            # habituation, i.e. updating intrinsic saliences
            self.int_sal = self.habituation(self.i_select, self.int_sal)
            if self.verbose:
                print 'Habituated intrinsic saliences:', self.int_sal

            # combining saliences
            self.tot_sal = self.nov_sal + self.int_sal
            if self.verbose:
                print 'Combined saliences:', self.tot_sal

            # record variables
            self.data = self.record()

        if self.verbose:
            print 'Run complete.'
        if self.output:
            self.outputfile_txt.close()

        return self.data





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='turn on verbose mode')
    parser.add_argument('-o', '--output', action='store_true', help='turn on file output mode')
    parser.add_argument('-T', '--T_max', nargs='?', type=int, default=10, help='total number of time steps')
    args = parser.parse_args()
    verbose = args.verbose
    output = args.output
    T_max = args.T_max

    myAgent = Agent(T_max=T_max, verbose=verbose, output=output)
    myAgent.run()
