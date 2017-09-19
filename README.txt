model2.py, v1.1.2, 17/09/19, by Max Murakami
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
    - During construction, specify:
        - either initial intrinsic action saliences (initial_int_sal), initial 
            prediction variables (initial_probs), or both.
            Both initial_int_sal and initial_probs must be 1 dimensional float ndarrays.
            Their length determines the number of actions (lengths must match if both
            are specified).
        - response_probs: the probabilities for each action to trigger a contingent environment
            response
            -> ndarray with shape (N_actions,) and float elements in [0;1]
        - triggered_int_sal: if specified, contains for each contingent response new intrinsic
            salience values for each action
            -> ndarray with shape (N_actions, N_action) and float elements
    - run() the object. Simulation data are returned as dictionary.
    - Free parameters:
        - exploration_rate: the higher, the more likely the agent executes actions
            with low saliences (softmax exploration)
        - learning_speed: the higher, the faster the agent acquires the contingencies
            -> time constant of novelty salience
        - hab_slowness: the higher, the less intrinsic saliences decrease due to habituation
            -> time constant of intrinsic salience
            (accepts different values for each action, specify as ndarray)
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
    - 1.1.2:
        - fixed bugs in output and verbose modes
    - 1.1.1:
        - triggered_int_sal can now contain negative elements for actions whose intrinsic 
            saliences should not be updated
    - 1.1:
        - made environment response to actions more general
            -> replaced failure_rate by response_probs, which govern contingency probabilities
                for each action
        - environment response can now trigger changes in intrinsic action saliences
            -> added triggered_int_sal, which govern intrinsic saliences as result of contingent
                response to specific actions
    - 1.0.3:
        - added failure_rate
    - 1.0.2.2:
        - hab_slowness argument of constructor now accepts ndarray with different
            values for each action
    - 1.0.2.1:
        - removed reinit() method
    - 1.0.2:
        - run() now returns a deep copy of data dict
    - 1.0.1:
        - added reinit() method
        - record() and run() now always return data dict, independent of --output
