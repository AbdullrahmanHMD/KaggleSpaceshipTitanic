# Imports:

# PyTorch imports:
import torch.nn as nn

# Model Factory imports:
from module_factory import ModuleFactory

# Collections imports:
from collections import OrderedDict

# Other imports:
import oyaml
import os


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model1_config.yaml')
MODULE_FACTORY = ModuleFactory()

class Model(nn.Module):
    def __init__(self, in_features, num_classes, model_config_path=DEFAULT_CONFIG_PATH):
        super(Model, self).__init__()


        self.model_config_path = model_config_path

        self.config = self.load_model_config()


        layer_names = list(self.config.keys())

        module_list = []
        for layer_name, modules in self.config.items():

            if layer_name == 'input':
                modules['Linear']['in_features'] = in_features
                module = MODULE_FACTORY.create_module(modules)

            elif layer_name == 'classification':
                modules['Linear']['out_features'] = num_classes
                module = MODULE_FACTORY.create_module(modules)

            else:
                module = MODULE_FACTORY.create_module(modules)

            module_list.append(module)

        self.model = nn.ModuleDict(list(zip(layer_names, module_list)))

        self.initialize_weights()


    def initialize_weights(self):
        for module in self.model.values():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)


    def load_model_config(self):
        """
        Loads the configuration file of the model.

        """
        with open(self.model_config_path, 'r') as file:
            config = oyaml.safe_load(file)

        return config


    def forward(self, x):
        for module in self.model.values():
            x = module(x)

        return x