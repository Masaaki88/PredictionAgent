model2.py, v1.0, 17/06/30, by Max Murakami
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