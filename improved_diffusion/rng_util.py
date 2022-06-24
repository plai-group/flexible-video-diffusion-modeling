import random
import torch as th
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    th.manual_seed(seed+1)
    th.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)


def get_random_state():
    return {
        "python": random.getstate(),
        "torch": th.get_rng_state(),
        "cuda": th.cuda.get_rng_state_all(),
        "numpy": np.random.get_state()
    }


def set_random_state(state):
    random.setstate(state["python"])
    th.set_rng_state(state["torch"])
    th.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


class RNG():

    def __init__(self, seed=None, state=None):

        self.state = get_random_state()
        with self:
            if seed is not None:
                set_random_seed(seed)
            elif state is not None:
                set_random_state(state)

    def __enter__(self):
        self.external_state = get_random_state()
        set_random_state(self.state)

    def __exit__(self, *args):
        self.state = get_random_state()
        set_random_state(self.external_state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class rng_decorator():

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            with RNG(self.seed):
                return f(*args, **kwargs)

        return wrapped_f 