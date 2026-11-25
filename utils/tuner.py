import csv
import itertools
import pathlib
from dataclasses import dataclass
from collections.abc import Callable
import functools
import inspect
import torch
import gc
from triton.runtime.errors import OutOfResources, PTXASError
from utils.measure import benchmark


DEFAULT_SKIPPED_ERRORS = (OutOfResources, PTXASError) # constant with default errors for triton and cute 

@dataclass(frozen=True)
class Param:
    options: tuple[int, ...]
    default: int | None = None

# Tunes and persists kernel hyperparameters to disk in a .csv file. Cols are arch, *keys, *params.
# Forgetting to invalidate tuning after modifying a kernel can result in inaccurate tunings. 
class Tuner:
    # feature ideas: allow passing in a search-space pruning function, add auto-invalidation based on kernel hash, allow args2keys to be a list and rename to key_source
    def __init__(self, fn, path, params: dict[str, Param], args2keys: Callable | None = None, errors_to_skip=DEFAULT_SKIPPED_ERRORS, *, invalidate_cache=False, autotune=True, benchmark_settings=None):
        self.fn = fn
        self.args2keys = args2keys if args2keys is not None else (lambda: {})
        self._fn_sig = inspect.signature(fn)
        self._args2keys_params = inspect.signature(self.args2keys).parameters
       
        self.path = pathlib.Path(path)
        self.params = params
        self.configs = {}
        self.benchmark_settings = {} if benchmark_settings is None else benchmark_settings
        self.keys = None
        self.errors_to_skip = errors_to_skip
        self.names = tuple(params.keys())
        self.autotune = autotune
        self.defaults = {name: self.params[name].default for name in self.names if self.params[name].default is not None}
        functools.update_wrapper(self, fn)

        invalid_params = set(self._args2keys_params) - set(self._fn_sig.parameters)
        if invalid_params:
            raise ValueError(f"args2keys has arguments not in tuned function signature: {invalid_params}")

        if not self.path.exists():
            return

        # everything is cleared bc a kernel changes typically affects all archs 
        # might have to include an all_archs flag for if a kernel modifies in an arch specific way
        if invalidate_cache: 
            return

        # failed parsing overwrites prev tuning bc kernel change is assumed 
        with self.path.open(newline="") as f:

            r = csv.DictReader(f)
            if r.fieldnames is None:
                return

            keys = tuple(col for col in r.fieldnames if col not in self.names)
            if not keys or keys[0] != "arch" or any(name not in r.fieldnames for name in self.names):
                return
            self.keys = keys
                

            for row in r:
                row = {key: int(value) for key, value in row.items()}
                self.configs[tuple(row[key] for key in self.keys)] = {name: row[name] for name in self.names}

    def __call__(self, *args, **kwargs):
        explicit = {name: kwargs.pop(name) for name in self.names if name in kwargs}
        keys = self._prep_keys(*args, **kwargs)
        if keys not in self.configs and self.autotune:
            self.gen_configs(*args, **kwargs)
        tuned = self.configs.get(keys, self.defaults) 

        return self.fn(*args, **kwargs, **(tuned | explicit))
        
    @staticmethod
    def _arch():
        arch = torch.cuda.get_device_capability()
        return arch[0] * 10 + arch[1]
        
    def _prep_keys(self, *inputs, **kwargs):
        bound_args = self._fn_sig.bind_partial(*inputs, **kwargs)
        bound_args.apply_defaults()

        keys = self.args2keys(**{param: bound_args.arguments[param] for param in self._args2keys_params})

        for key, value in keys.items():
            if not isinstance(value, int):
                raise TypeError(f"key {key} is not an int")
            keys[key] = int(value)
        
        if self.keys is None:
            self.keys = ("arch", *keys)
            collisions = set(self.keys) & set(self.names)
            assert not collisions, f"params and keys name collision: {collisions}"

        if set(keys) != set(self.keys[1:]):
            raise ValueError(f"expected keys {self.keys[1:]}, got {tuple(keys)}")

        return (self._arch(), *(keys[k] for k in self.keys[1:]))

    def save_configs(self):
        if self.keys is None:
            return
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            columns = (*(self.keys), *self.names)
            w.writerow(columns)

            for key in sorted(self.configs):
                w.writerow((*key, *(self.configs[key][name] for name in self.names)))
                

    def clear_configs(self, all_archs=False):
        arch = self._arch()
        
        if all_archs:
            self.configs = {}
        else:
            self.configs = {keys: value for keys, value in self.configs.items() if keys[0] != arch}     

        self.save_configs()

    
    def gen_configs(self, *args, autosave_config=True, **kwargs):

        keys = self._prep_keys(*args, **kwargs)

        fastest = (None, float('inf'))
        all_options = (param.options for param in self.params.values())
        for cfg in itertools.product(*all_options):
            config = dict(zip(self.names, cfg))
            try:
                measured_time = benchmark(self.fn, *args, **(kwargs | config), **self.benchmark_settings)
                if measured_time < fastest[1]:
                    fastest = (config, measured_time)
            except self.errors_to_skip:
                continue
            finally:
                gc.collect() # exceptions

        if fastest[0] is None:
            raise RuntimeError(f"no config ran for {keys}")
        self.configs[keys] = fastest[0]

        if autosave_config:
            self.save_configs()

        return self.configs[keys] 
   
    
def tuner(path, params: dict[str, Param], args2keys: Callable, errors_to_skip=DEFAULT_SKIPPED_ERRORS, invalidate_cache=False, autotune=True, benchmark_settings=None):
    return lambda fn: Tuner(fn, path, params, args2keys, errors_to_skip, invalidate_cache=invalidate_cache, autotune=autotune, benchmark_settings=benchmark_settings)

        