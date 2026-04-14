import math
from collections import namedtuple

EpsDelta = namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

class SimpleAccountant:

    def __init__(self):
        self.eps_sum = 0.0
        self.delta_sum = 0.0

    def accumulate_privacy_spending(self, eps_delta):
        eps, delta = eps_delta
        self.eps_sum += eps
        self.delta_sum += delta

    def get_privacy_spent(self):
        return [EpsDelta(self.eps_sum, self.delta_sum)]
