from importlib import util as import_util
import os
import sys


class ImportTool(object):
    # @staticmethod
    # def import_module(module_dir, module_name: str):
    #     if module_name.endswith('.py'):
    #         module_name = module_name.replace('.py', '')
    #
    #     if module_dir not in sys.path:
    #         sys.path.append(module_dir)
    #     load_module = import_module(module_name)
    #     return load_module

    @staticmethod
    def import_module(module_dir, module_name: str):
        module_path = os.path.join(module_dir, module_name)

        # Load the module from the given path
        spec = import_util.spec_from_file_location('module_name', module_path)
        module = import_util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Add the module to sys.modules
        module_name_base = module_name.replace('.py', '')
        sys.modules[module_name_base] = module

        return module

    @staticmethod
    def get_module_names(module):
        return dir(module)

    @staticmethod
    def get_module_function(module, name):
        if name not in ImportTool.get_module_names(module):
            raise ValueError("[ERROR]: Non-Valid Module Function")
        return getattr(module, name)

    @staticmethod
    def remove_module(module_name):
        sys.modules.pop(module_name)
