import numpy as np
from decimal import *  # Floating point errors in ML are bad, mkay?


class PerceptionModel:

    def __init__(self, weights, learning_coefficient, threshold, func, debug=False):
        self.target_func = func
        self.learning_coefficient = Decimal(str(learning_coefficient))

        self.weights = []
        for w in weights:
            self.weights.append(Decimal(str(w)))

        self.threshold = threshold
        self.trained = False
        self.debug = debug

        # This is honestly easier to just precalculate and save right now, though this obviously won't scale super well
        self.input_sets = []
        inputs = 2 ** len(weights)
        for i in range(inputs):
            variable_count = len(weights)
            in_set = list(bin(i)[2:].zfill(variable_count))
            in_set = np.array(in_set)
            s = []
            for j in in_set:
                s.append(Decimal(j))
            self.input_sets.append(s)

    def get_weights(self):
        s = []
        for w in self.weights:
            s.append(str(w))
        return str(s)

    def train(self, max_iterations=0):
        if self.debug:
            print(f"{'epoch':<10} {'Old Weights':<20} {'New Weights':<30}")

        done = False
        iteration = 0
        while not done:
            old_weights = self.get_weights()
            iteration += 1
            done = self.epoch()
            new_weights = self.get_weights()
            if not done and self.debug:
                print(f"{iteration:<10} {old_weights:<20} {new_weights:<30}")
            if max_iterations != 0 and iteration >= max_iterations:
                break

        if done:
            self.trained = True

        if self.debug:
            print("Final Weights: " + str(self.get_weights()))

    def epoch(self):
        done = True
        for attributes in self.input_sets:

            result = self.check(attributes)
            target = self.target_func(attributes)

            if target != result:
                done = False

            for i in range(len(attributes)):
                adj_weight = self.weights[i] + self.learning_coefficient * Decimal(str(target - result)) * attributes[i]
                self.weights[i] = adj_weight

        return done

    '''Internal function to handle checking a single input'''
    def check(self, inputs):
        products = np.multiply(self.weights, inputs)
        total = np.sum(products)
        result = total > self.threshold
        if result:
            return 1
        return 0
