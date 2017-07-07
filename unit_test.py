'''
unit test for model2 v1.0.2.2, 17/07/06, by Max Murakami
'''

import numpy as np
import unittest
import model2
import argparse

# this determines the verbosity of the Agent's methods during testing
# -> use command line flag -v to turn on verbosity of the unit test itself
VERBOSE = False

class TestAgentClass(unittest.TestCase):

    #### Constructor

    def test_constructor_args(self):
        # test types of Agent constructor arguments:
        #   initial_int_sal, initial_probs, exploration_rate
        # Agent has 4 actions by default
        
        # bad initial_int_sal values
        initial_int_sal_args = [2.2, False, '2', (2,3), [], np.array([]), ['d'], np.array(['s']), np.array([True])]
        for arg in initial_int_sal_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(initial_int_sal=arg, verbose=VERBOSE)

        # bad initial_probs values
        initial_probs_args = [0, 0.5, False, '0', (0.5, 0.5), [], [-0.5], [2.], np.array([]), np.array(['3']), np.array([True])]
        for arg in initial_probs_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(initial_probs=arg, verbose=VERBOSE)

        # bad combination of initial_int_sal and initial_probs
        with self.assertRaises(AssertionError):
            MyAgent = model2.Agent(initial_int_sal=np.arange(3.), initial_probs=np.zeros([5]), verbose=VERBOSE)

        # bad exploration_rate arguments
        exploration_rate_args = [False, '2.', [2.], (2.,3.), np.array(2.), 0.0]
        for arg in exploration_rate_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(exploration_rate=arg, verbose=VERBOSE) 

        # bad learning_speed arguments
        learning_speed_args = [False, '0.5', [0.5], (0.4, 0.2), np.array(0.4), 0.0, 1.0]
        for arg in learning_speed_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(learning_speed=arg, verbose=VERBOSE)

        # bad hab_slowness arguments
        hab_slowness_args = [False, '0.5', [0.5], (0.4, 0.2), np.array(0.4), 0.0, 1.0, np.array([0.0,0.2]),
            np.array([0.1,1.0])]
        for arg in hab_slowness_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(initial_int_sal=np.arange(2.), hab_slowness=arg, verbose=VERBOSE)

        # bad failure_rate arguments
        learning_speed_args = [False, '0.5', [0.5], (0.4, 0.2), np.array(0.4), -0.1, 2.0]
        for arg in learning_speed_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(failure_rate=arg, verbose=VERBOSE)

        # bad T_max arguments
        T_max_args = ['2', [2], (2,2), np.array([2]), False, 2.0]
        for arg in T_max_args:
            with self.assertRaises(AssertionError):
                MyAgent = model2.Agent(T_max=arg, verbose=VERBOSE) 

       

    #### select_action()

    def test_select_action_args(self):
        # test types of select_action method arguments

        MyAgent = model2.Agent(initial_int_sal=np.arange(3.), verbose=VERBOSE)

        # bad saliences arguments
        args = ['string', 4, 2.2, (2.3), False, [], np.array([]),
            np.array(['2.','3.','1.']), np.array([[2.,3.,1.]]), np.array([(2.,3.,1.)]),
            np.array([False,True,False]), np.array([2.,3.])]
        for arg in args:
            with self.assertRaises(AssertionError):
                MyAgent.select_action(arg)


    def test_select_action_output(self):
        # test output of select_action method

        MyAgent = model2.Agent(initial_int_sal=np.arange(10.), exploration_rate=1e3, verbose=VERBOSE)
        saliences = np.arange(10.)
        i_selected_action = MyAgent.select_action(saliences)

        # test generic properties
        self.assertIsInstance(i_selected_action, int)
        self.assertIsNotNone(i_selected_action)
        self.assertIn(i_selected_action, range(10))

        # test special case of greedy action selection
        MyAgent = model2.Agent(initial_int_sal=np.arange(2.), exploration_rate=1e-3, verbose=VERBOSE)
        saliences = np.array([0.2, 0.1])
        i_selected_action = MyAgent.select_action(saliences)

        self.assertEqual(i_selected_action, 0)




    #### environment_response()

    def test_environment_response_args(self):
        # test types and values of environment_response method arguments

        MyAgent = model2.Agent(initial_int_sal=np.arange(3.), verbose=VERBOSE)

        # bad i_action arguments
        i_action_args = ['0', [0], np.array([0]), False, 0.0, -1, 11]
        for arg in i_action_args:
            with self.assertRaises(AssertionError):
                MyAgent.environment_response(arg)


    def test_environment_response_output(self):
        # test output of environment_response method

        """
        MyAgent = model2.Agent(initial_int_sal=np.arange(10.), verbose=VERBOSE)
        for i_action in xrange(10):
            response = MyAgent.environment_response(i_action)

            self.assertIsNotNone(response)
            self.assertIsInstance(response, int)
            self.assertIn(response, range(2))
        """
        MyAgent = model2.Agent(initial_int_sal=np.arange(10.), failure_rate=1.0,
            verbose=VERBOSE)
        for i_action in xrange(10):
            response = MyAgent.environment_response(i_action)

            self.assertIsNotNone(response)
            self.assertIsInstance(response, int)
            self.assertEqual(response, 0)



    #### update_predictions()
    def test_update_predictions_args(self):
        # test types and values of update_predictions method arguments

        MyAgent = model2.Agent(initial_int_sal=np.arange(3.), verbose=VERBOSE)
        i_action = 0
        env_response = 0
        probs = np.zeros([3])

        # bad i_action arguments
        i_action_args = ['0', [0], np.array([0]), False, 0.0, (0,0), -1, 3]
        for arg in i_action_args:
            with self.assertRaises(AssertionError):
                MyAgent.update_predictions(arg, env_response, probs)

        # bad env_response arguments
        env_response_args = ['0', [0], np.array([0]), False, (0,0), 0.0, -1, 2]
        for arg in env_response_args:
            with self.assertRaises(AssertionError):
                MyAgent.update_predictions(i_action, arg, probs)

        # bad probs arguments
        probs_args = ['string', 0, (0.0,0.0,0.0), [0.0,0.0,0.0], np.array([0.0]),
            np.array([0.0,0.0,-0.1]), np.array([0.0,0.0,1.1]), np.array([0,0,0]),
            False, np.array([False,False,False]), np.array(['0.0','0.0','0.0']),
            np.array([[0.0,0.0,0.0]])]
        for arg in probs_args:
            with self.assertRaises(AssertionError):
                MyAgent.update_predictions(i_action, env_response, arg)


    def test_update_predictions_output(self):
        # test output of update_predictions method

        MyAgent = model2.Agent(initial_int_sal=np.arange(2.), learning_speed=0.5, verbose=VERBOSE)
        probs = np.array([0.0, 1.0])
        for i in xrange(1,10):
            probs = MyAgent.update_predictions(0, 1, probs)
            
            # check generic properties
            self.assertIsNotNone(probs)
            self.assertIsInstance(probs, np.ndarray)
            self.assertEqual(probs.ndim, 1)
            self.assertEqual(len(probs), 2)
            self.assertTrue(np.isfinite(probs).all())
            self.assertEqual(probs.dtype, float)

            # check specific values
            self.assertAlmostEqual(probs[0], 1.0-(0.5**i))
            self.assertAlmostEqual(probs[1], 1.0)

        probs = np.array([0.0, 1.0])
        for i in xrange(1,10):
            probs = MyAgent.update_predictions(1, 0, probs)
        
            # check generic properties
            self.assertIsNotNone(probs)
            self.assertIsInstance(probs, np.ndarray)
            self.assertEqual(probs.ndim, 1)
            self.assertEqual(len(probs), 2)
            self.assertTrue(np.isfinite(probs).all())
            self.assertEqual(probs.dtype, float)

            # check specific values
            self.assertAlmostEqual(probs[0], 0.0)
            self.assertAlmostEqual(probs[1], 0.5**i)




    #### get_novelty_saliences()

    def test_get_novelty_saliences_args(self):
        # test types and values of get_novelty_saliences method arguments

        MyAgent = model2.Agent(initial_int_sal=np.arange(3.), verbose=VERBOSE)

        # bad probs arguments
        probs_args = ['string', 4, 2.2, (2.3), False, [], np.array([]),
            np.array(['2.', '3.', '1.']), np.array([[0.1,0.1,0.1]]), np.array([(0.1,0.1,0.1)]),
            np.array([False,True,False]), np.array([0.1,0.1]), np.array([-0.2, 0.2, 0.2]),
            np.array([1.1, 0.2, 0.2]), np.array([[0.2, 0.2, 0.2]])]
        for arg in probs_args:
            with self.assertRaises(AssertionError):
                MyAgent.get_novelty_saliences(arg)


    def test_get_novelty_saliences_output(self):
        # test output of get_novelty_saliences method

        MyAgent = model2.Agent(initial_int_sal=np.arange(11.), verbose=VERBOSE)
        probs = np.arange(0.0, 1.1, 0.1)
        nov_saliences = MyAgent.get_novelty_saliences(probs)

        # check generic properties
        self.assertIsInstance(nov_saliences, np.ndarray)
        self.assertEqual(nov_saliences.dtype, float)
        self.assertEqual(nov_saliences.ndim, 1)
        self.assertTrue(np.isfinite(nov_saliences).all())
        self.assertTrue((nov_saliences >= 0.0).all())

        # check specific values
        expected_values = [0.0, 0.234497796795, 0.360964047444, 0.440645449615, 0.485475297227,
            0.5, 0.485475297227, 0.440645449615, 0.360964047444, 0.234497796795, 0.0]
        self.assertEqual(len(nov_saliences), 11)
        for i in xrange(11):
            self.assertAlmostEqual(nov_saliences[i], expected_values[i])



    #### habituation()

    def test_habituation_args(self):
        # test types and values of habituation method argument
        
        MyAgent = model2.Agent(initial_int_sal=np.arange(3.), verbose=VERBOSE)

        # bad i_select arguments
        i_select_args = ['0', [0], (0,0), np.array([0]), 0.0, False, -1, 3]
        for arg in i_select_args:
            with self.assertRaises(AssertionError):
                MyAgent.habituation(arg, np.arange(3.))

        # bad int_sal arguments
        int_sal_args = ['1.0, 2.0', [1.0, 2.0], (1.0, 2.0), 1.0, np.array([1,2,3]),
            np.array([False, False, False]), np.array(['1.0', '2.0', '3.0']),
            np.array([1.0, 2.0]), np.array([-1.0, 1.0, 2.0])]
        for arg in int_sal_args:
            with self.assertRaises(AssertionError):
                MyAgent.habituation(0, arg)


    def test_habituation_output(self):
        # test output of habituation method

        hab_slowness = np.array([0.5, 0.8])

        MyAgent = model2.Agent(initial_int_sal=np.arange(2.), hab_slowness=hab_slowness, verbose=VERBOSE)

        int_sal = np.array([1.0, 0.7])
        int_sal = MyAgent.habituation(0, int_sal)

        # check generic properties
        self.assertIsNotNone(int_sal)
        self.assertIsInstance(int_sal, np.ndarray)
        self.assertEqual(int_sal.ndim, 1)
        self.assertEqual(len(int_sal), 2)
        self.assertTrue(np.isfinite(int_sal).all())
        self.assertEqual(int_sal.dtype, float)

        # check specific values
        self.assertAlmostEqual(int_sal[0], 1.0*0.5)
        self.assertAlmostEqual(int_sal[1], 0.7)

        for i in xrange(1,10):
            int_sal = MyAgent.habituation(1, int_sal)

            # check generic properties
            self.assertIsNotNone(int_sal)
            self.assertIsInstance(int_sal, np.ndarray)
            self.assertEqual(int_sal.ndim, 1)
            self.assertEqual(len(int_sal), 2)
            self.assertTrue(np.isfinite(int_sal).all())
            self.assertEqual(int_sal.dtype, float)

            # check specific values
            self.assertAlmostEqual(int_sal[0], 0.5)
            self.assertAlmostEqual(int_sal[1], 0.7*(0.8**i))



    #### record()

    def test_record_output(self):
        # test output of record method

        T_max = 10
        exploration_rate = 2.0
        learning_speed = 0.8
        hab_slowness = np.array([0.9, 0.7])

        MyAgent = model2.Agent(initial_int_sal=np.arange(2.), initial_probs=np.array([0.55,0.66]),
            T_max=T_max, exploration_rate=exploration_rate, learning_speed=learning_speed,
            hab_slowness=hab_slowness, verbose=VERBOSE)

        MyAgent.t = 2
        MyAgent.i_select = 1
        MyAgent.response = 1
        MyAgent.int_sal = np.array([0.4, 0.3])
        MyAgent.nov_sal = np.array([0.3, 0.2])
        MyAgent.tot_sal = np.array([0.2, 0.1])
        MyAgent.probs = np.array([0.1, 0.0])

        data = MyAgent.record()

        # check generic properties
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)

        keys = ['exploration_rate', 'learning_speed', 'hab_slowness', 'i_select',
            'env_response', 'int_sal', 'nov_sal', 'tot_sal', 'probs', 'failure_rate']
        data_keys = data.keys()        

        self.assertEqual(len(data_keys), len(keys))  
        self.assertItemsEqual(data_keys, keys)

        self.assertIsNotNone(data['hab_slowness'])
        self.assertIsInstance(data['hab_slowness'], np.ndarray)
        self.assertEqual(data['hab_slowness'].dtype, float)
        self.assertEqual(data['hab_slowness'].ndim, 1)
        self.assertEqual(len(data['hab_slowness']), 2)
        for key in ['exploration_rate', 'learning_speed']:
            self.assertIsNotNone(data[key])
            self.assertIsInstance(data[key], float)
            self.assertGreater(data[key], 0.0)
        for key in ['i_select', 'env_response']:
            self.assertIsNotNone(data[key])
            self.assertIsInstance(data[key], np.ndarray)
            self.assertEqual(len(data[key]), T_max)
            self.assertEqual(data[key].ndim, 1)
            self.assertTrue(np.isfinite(data[key]).all())
            self.assertEqual(data[key].dtype, int)
            self.assertTrue((data[key] > -1).all())
            self.assertTrue((data[key] < 2).all())
        for key in ['int_sal', 'nov_sal', 'tot_sal', 'probs']:
            self.assertIsNotNone(data[key])
            self.assertIsInstance(data[key], np.ndarray)
            self.assertEqual(data[key].ndim, 2)
            self.assertEqual(len(data[key]), 2)
            self.assertEqual(len(data[key][0]), T_max)
            self.assertEqual(data[key].dtype, float)
            self.assertTrue(np.isfinite(data[key]).all())
            self.assertTrue((data[key] >= 0.0).all())

        # check specific values
        self.assertAlmostEqual(data['exploration_rate'], exploration_rate)
        self.assertAlmostEqual(data['learning_speed'], learning_speed)
        self.assertAlmostEqual(data['hab_slowness'][0], hab_slowness[0])
        self.assertAlmostEqual(data['hab_slowness'][1], hab_slowness[1])
        self.assertEqual(data['i_select'][2], 1)
        self.assertEqual(data['env_response'][2], 1)
        self.assertAlmostEqual(data['int_sal'][0][2], 0.4)
        self.assertAlmostEqual(data['int_sal'][1][2], 0.3)
        self.assertAlmostEqual(data['nov_sal'][0][2], 0.3)
        self.assertAlmostEqual(data['nov_sal'][1][2], 0.2)
        self.assertAlmostEqual(data['tot_sal'][0][2], 0.2)
        self.assertAlmostEqual(data['tot_sal'][1][2], 0.1)
        self.assertAlmostEqual(data['probs'][0][2], 0.1)
        self.assertAlmostEqual(data['probs'][1][2], 0.0)



    #### run()

    def test_run_output(self):
        # test output of run method

        T_max = 2
        exploration_rate = 2.0
        learning_speed = 0.8
        hab_slowness = np.array([0.9, 0.7])
        initial_int_sal = np.array([0.0, 1.0])
        initial_probs = np.array([0.55, 0.66])

        MyAgent = model2.Agent(initial_int_sal=initial_int_sal,
            initial_probs=initial_probs, T_max=T_max,
            exploration_rate=exploration_rate, learning_speed=learning_speed,
            hab_slowness=hab_slowness, verbose=VERBOSE)

        data = MyAgent.run()


        # check generic properties
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)

        test_keys = ['exploration_rate', 'learning_speed', 'hab_slowness', 'i_select',
            'env_response', 'int_sal', 'nov_sal', 'tot_sal', 'probs', 'failure_rate']
        data_keys = data.keys()

        self.assertIsNotNone(data_keys)
        self.assertIsInstance(data_keys, list)
        self.assertEqual(len(data_keys), len(test_keys))
        self.assertItemsEqual(data_keys, test_keys)
        
        self.assertIsNotNone(data['hab_slowness'])
        self.assertIsInstance(data['hab_slowness'], np.ndarray)
        self.assertEqual(data['hab_slowness'].dtype, float)
        self.assertEqual(data['hab_slowness'].ndim, 1)
        self.assertEqual(len(data['hab_slowness']), 2)
        for key in ['exploration_rate', 'learning_speed']:
            self.assertIsNotNone(data[key])
            self.assertIsInstance(data[key], float)
            self.assertGreater(data[key], 0.0)
        for key in ['i_select', 'env_response']:
            self.assertIsNotNone(data[key])
            self.assertIsInstance(data[key], np.ndarray)
            self.assertEqual(len(data[key]), T_max)
            self.assertEqual(data[key].ndim, 1)
            self.assertTrue(np.isfinite(data[key]).all())
            self.assertEqual(data[key].dtype, int)
            self.assertTrue((data[key] > -1).all())
            self.assertTrue((data[key] < 2).all())
        for key in ['int_sal', 'nov_sal', 'tot_sal', 'probs']:
            self.assertIsNotNone(data[key])
            self.assertIsInstance(data[key], np.ndarray)
            self.assertEqual(data[key].ndim, 2)
            self.assertEqual(len(data[key]), 2)
            self.assertEqual(len(data[key][0]), T_max)
            self.assertEqual(data[key].dtype, float)
            self.assertTrue(np.isfinite(data[key]).all())
            self.assertTrue((data[key] >= 0.0).all())


        # check specific values
        self.assertAlmostEqual(data['exploration_rate'], exploration_rate)
        self.assertAlmostEqual(data['learning_speed'], learning_speed)
        self.assertAlmostEqual(data['hab_slowness'][0], hab_slowness[0])
        self.assertAlmostEqual(data['hab_slowness'][1], hab_slowness[1])
        


if __name__ == '__main__':
    unittest.main()
