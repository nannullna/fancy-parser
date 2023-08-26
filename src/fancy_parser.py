from typing import Dict, List, Tuple, Union, Optional, Literal, Mapping, Iterable, Callable, Any, get_type_hints
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace, ArgumentError
import sys
import dataclasses
from dataclasses import fields
from inspect import isclass
from enum import Enum

import json
import yaml

SUPPRESS = '==SUPPRESS=='
NoneType = type(None)

# Will support colors in the future
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class FancyParser(ArgumentParser):
    """A subclass of ArgumentParser that allows for dataclass argument and yaml/json configuration file parsing.
    Most implementations are based on HuggingFace's transformers library's HfArgumentParser.

    Please refer to https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/hf_argparser.py
    """
    def __init__(self, classes: Union[type, Iterable[type]], add_config_arg: bool=True, *args, **kwargs):
        """Initialize the FancyParser.

        Args:
            classes (`type` or `Iterable[type]`) 
                The dataclass(es) to parse arguments into.
            add_config_arg (`bool`, *optional*): 
                Whether to add a `--config` argument to the parser. Defaults to True.
            args: 
                Positional arguments to pass to the ArgumentParser constructor.
            kwargs: 
                Keyword arguments to pass to the ArgumentParser constructor.
        """
        if kwargs.get('formatter_class', None) is None:
            kwargs['formatter_class'] = ArgumentDefaultsHelpFormatter
        
        super().__init__(*args, **kwargs)
        self.add_config_arg = add_config_arg
        
        if isinstance(classes, type):
            classes = [classes]
        self.classes = classes
        for cls in classes:
            self.add_arguments_from_dataclass(cls)

    @staticmethod
    def to_bool(s: Union[bool, str]) -> bool:
        """Convert an input to a boolean.

        Args:
            s (`bool` or `str`): The input to convert.
        
        Returns:
            `bool`: The converted boolean.
        """
        if isinstance(s, bool):
            return s
        if s.lower() in ['true', 't', '1', 'yes', 'y']:
            return True
        elif s.lower() in ['false', 'f', '0', 'no', 'n']:
            return False
        else:
            raise ValueError(f'Cannot convert {s} to a boolean.')
    
    @staticmethod
    def make_choice_type(choices: Iterable[Any]) -> Callable[[str], Any]:
        """Convert an input to a choice.

        Args:
            choices (`Iterable[Any]`): The choices to convert to. Must implement `__str__`.
        
        Returns:
            `Callable[[str], Any]`: The converter function that maps a string to a choice.
        """
        if isinstance(choices, Mapping):
            choices = {str(k): v for k, v in choices.items()}
        else:
            choices = {str(c): c for c in choices}
        return lambda s: choices.get(s, s)
    

    def add_arguments_from_dataclass(self, cls: type):
        """Add arguments to the parser from a dataclass.

        Args:
            cls (`type`): The dataclass to add arguments from.

        Raises:
            NotImplementedError: If the dataclass has more than one optional type.
        """
        parser = self.add_argument_group(cls.__name__)

        type_hints = get_type_hints(cls)
        for field in fields(cls):
            field.type = type_hints[field.name]
            try:
                self.add_argument_from_field(parser, field)
            except Exception as e:
                print(field)
                raise e


    def add_argument_from_field(self, parser: ArgumentParser, field: dataclasses.Field):
        """Add an argument to the parser from a dataclass field.
        
        Need to handle the followings:
        * Optional[X] and Union[X, NoneType]
        * Literal[X, Y, Z] and Enum
        * List[X] -- default_factory
        * bool
        * Primitive types (str, int, float)

        Args:
            parser (`ArgumentParser`): The parser to add the argument to.
            field (`dataclasses.Field`): The field to add the argument from.

        Raises:
            NotImplementedError: If the dataclass has more than one optional type.
        """
        kwargs = field.metadata.copy()

        # Handle Optional[X] and Union[X, NoneType], Union[NoneType, X]
        origin_type = getattr(field.type, '__origin__', field.type)
        if (origin_type is Optional) or ((origin_type is Union) and (NoneType in field.type.__args__)):
            if len(field.type.__args__) > 2:
                raise NotImplementedError('FancyParser does not support more than one optional type.')
            kwargs['type'] = field.type.__args__[0] if field.type.__args__[0] is not NoneType else field.type.__args__[1]
            kwargs['default'] = None if field.default is dataclasses.MISSING else field.default

        # Handle Literal[X, Y, Z]
        if origin_type is Literal:
            kwargs['choices'] = list(field.type.__args__)
            kwargs['type'] = self.make_choice_type(kwargs['choices'])
            if field.default is not dataclasses.MISSING:
                kwargs['default'] = field.type(field.default)
            else:
                kwargs['required'] = True
        
        # Handle Enum
        elif isinstance(origin_type, type) and issubclass(origin_type, Enum):
            kwargs['choices'] = [e for e in field.type]
            kwargs['type'] = self.make_choice_type({e.value: e for e in field.type})
            if field.default is not dataclasses.MISSING:
                kwargs['default'] = field.type(field.default)
            else:
                kwargs['required'] = True

        # Handle List[X] -- default_factory
        elif isclass(origin_type) and issubclass(origin_type, list):
            kwargs['type'] = field.type.__args__[0]
            kwargs['nargs'] = '*'
            if field.default_factory is not dataclasses.MISSING:
                kwargs['default'] = field.default_factory()
            elif field.default is not dataclasses.MISSING:
                kwargs['required'] = True

        # Handle bool
        elif field.type is bool:
            kwargs['type'] = self.to_bool
            kwargs['default'] = False if field.default is dataclasses.MISSING else field.default
            kwargs['nargs'] = '?'
            kwargs['const'] = True

        # Primitive types
        else:
            if field.default is not dataclasses.MISSING:
                kwargs['default'] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs['default'] = field.default_factory()
            else:
                kwargs['required'] = True
        
        # Add the argument
        parser.add_argument(f'--{field.name}', **kwargs)


    def set_defaults_from_config(self, config_file: str):
        """Set the default values of the parser from a config file.

        Args:
            config_file (`str`): The path to the config file.
        """
        if config_file.endswith('.json'):
            config_dict = self.parse_json(config_file, return_dict=True)
        elif config_file.endswith('.yaml'):
            config_dict = self.parse_yaml(config_file, return_dict=True)
        else:
            raise ValueError(f'Unsupported config file format: {config_file}')
        self.set_defaults(**config_dict)

    
    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()
        
        # parse config files
        if self.add_config_arg:
            temp_parser = ArgumentParser(add_help=False)
            temp_parser.add_argument('-c', '--config', type=str, nargs='*', help='Path to config file(s).')
            temp_namespace, args = temp_parser.parse_known_args(args=args)

            if temp_namespace.config is not None:
                for config_file in temp_namespace.config:
                    self.set_defaults_from_config(config_file)

            self.add_argument('-c', '--config', type=str, nargs='*', help='Path to config file(s).')

        return super().parse_known_args(args=args, namespace=namespace)
    

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_args: bool=False,
    ) -> Tuple:
        """Parse arguments into dataclasses.
        There are three ways to provide arguments:
          1. As default values in the dataclass. 
          2. As config file(s). 
          3. As command line arguments. 
        
        The order of precedence is 3 > 2 > 1, i.e., 
        command line arguments override config file arguments, 
        which override default values in the dataclass.
        
        Args:
            args (`List[str]`, *optional*): The arguments to parse. If not provided, will use sys.argv.
            return_remaining_args (`bool`, *optional*): Whether to return the remaining arguments after parsing.

        Returns:
            `Tuple[...]`: The parsed dataclasses.
        """
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for cls in self.classes:
            keys = {f.name for f in fields(cls) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in inputs.keys():
                delattr(namespace, k)
            obj = cls(**inputs)
            outputs.append(obj)
        
        if return_remaining_args:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f'Unused arguments: {remaining_args}')
            return (*outputs,)


    def parse_dict(self, args: Dict[str, Any], allow_extra_keys: bool=False) -> Tuple:
        """Parse a dictionary into dataclasses.

        Args:
            args (`Dict[str, Any]`): The dictionary to parse.
            allow_extra_keys (`bool`, *optional*): Whether to allow extra keys in the dictionary.

        Returns:
            `Tuple[...]`: The parsed dataclasses.
        """
        unused_keys = set(args.keys())
        outputs = []
        for cls in self.classes:
            keys = {f.name for f in fields(cls) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            for k in keys:
                unused_keys.discard(k)
            obj = cls(**inputs)
            outputs.append(obj)
        if unused_keys and not allow_extra_keys:
            if self.add_config_arg and ("config" in unused_keys):
                unused_keys.remove("config")
            raise ValueError(f'Unused keys: {unused_keys}')
        return (*outputs,)
    

    def parse_json(self, json_file: str, return_dict=False, allow_extra_keys: bool=False) -> Tuple:
        """Parse a json file into dataclasses.

        Args:
            json_file (`str`): The path to the json file to parse.
            allow_extra_keys (`bool`, *optional*): Whether to allow extra keys in the dictionary.

        Returns:
            `Tuple[...]`: The parsed dataclasses.
        """
        with open(json_file, encoding='utf=-8', mode='r') as f:
            args = json.load(f)
        if return_dict:
            return args
        return self.parse_dict(args, allow_extra_keys=allow_extra_keys)
    

    def parse_yaml(self, yaml_file: str, return_dict=False, allow_extra_keys: bool=False) -> Tuple:
        """Parse a yaml file into dataclasses.

        Args:
            yaml_file (`str`): The path to the yaml file to parse.
            allow_extra_keys (`bool`, *optional*): Whether to allow extra keys in the dictionary.
        
        Returns:
            `Tuple[...]`: The parsed dataclasses.
        """
        with open(yaml_file, encoding='utf=-8', mode='r') as f:
            args = yaml.safe_load(f)
        if return_dict:
            return args
        return self.parse_dict(args, allow_extra_keys=allow_extra_keys)

