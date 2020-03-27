import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import itertools as it

# TODO(Oleguer): Think about the structure of all this

class MetaParamOptimizer:
    def __init__(self, save_path=""):
        self.save_path = save_path  # Where to save best result and remaining to explore
        pass

    def grid_search(self, evaluator, search_space, fixed_args):
        """ Performs grid search on specified search_space
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached, parameters that obtained it
        """
        points_to_evaluate = self.__get_all_dicts(search_space)
        max_value = -float("inf")
        max_result = None
        max_params = None
        for indx, evaluable_args in enumerate(points_to_evaluate):
            print("GridSearch evaluating:", indx, "/", len(points_to_evaluate), ":", evaluable_args)
            args = {**evaluable_args, **fixed_args}  # Merge kwargs and evaluable_args dicts
            result = evaluator(**args)
            if result["value"] > max_value:
                max_value = result["value"]
                max_result = result
                max_params = evaluable_args
        return max_result, max_params

    def GP_optimizer(self, evaluator, search_space, fixed_args):
        pass # The other repo

    def __get_all_dicts(self, param_space):
        """ Given:
            dict of item: list(elems)
            returns:
            list (dicts of item : elem)
        """
        allNames = sorted(param_space)
        combinations = it.product(*(param_space[Name] for Name in allNames))
        dictionaries = []
        for combination in combinations:
            dictionary = {}
            for indx, name in enumerate(allNames):
                dictionary[name] = combination[indx]
            dictionaries.append(dictionary)
        return dictionaries