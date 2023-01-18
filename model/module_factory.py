# Imports:
#
# PyTorch imports:
import torch
import torch.nn as nn


class ModuleFactory():

    def create_module(self, module_dict):
        """
        Creates a sequential (torch.nn.Sequential) module given a dictionary
        with keys being module names and values being the parameters of
        these modules.

        Parameters:
        -----------
        module_dict (dict):
            The dictionary that contains the module names as keys and their
            parameters as values.

        Returns:
            final_module (torch.nn.Sequential):
                A PyTorch module that contains the modules specified in the
                module_format dictionary.

        """
        # Checking if the dictionary is empty and throwing an exception if
        # it is.
        if len(module_dict.items()) == 0:
            raise ValueError("module_format should not be empty")

        # Will hold all the created modules.
        module_list = []

        # Iterating over the modules and their parameters:
        for module_name, params in module_dict.items():
            if module_name == 'Linear':
                module = nn.Linear(**params)

            elif module_name == 'BatchNorm1d':
                module = nn.BatchNorm1d(**params)

            elif module_name == 'Dropout':
                module = nn.Dropout(**params)

            elif module_name == 'LeakyReLU':
                module = nn.LeakyReLU(**params)

            module_list.append(module)

        # Creating the Sequential module that contains all the modules
        # specified in the module_format parameter:
        final_module = nn.Sequential(*module_list)

        return final_module