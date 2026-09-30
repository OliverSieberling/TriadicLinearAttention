"""nanoGPT-style configurator, exec()'d by train.py (not a module):
$ python train.py config/override_file.py --batch_size=32
runs config/override_file.py first, then overrides batch_size in globals().
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        _cfg_before = set(globals())
        exec(open(config_file).read())
        _cfg_bad = {k for k in set(globals()) - _cfg_before
                    if not k.startswith('_') and k not in config_keys and not callable(globals()[k])}
        assert not _cfg_bad, f"config file {config_file} sets unknown key(s): {sorted(_cfg_bad)}"
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)  # bool/number/tuple/etc
            except (SyntaxError, ValueError):
                attempt = val  # fall back to the raw string
            # a None default carries no type, so any value is accepted (ffn_intermediate_size)
            assert globals()[key] is None or type(attempt) == type(globals()[key]), \
                f"{key}: expected {type(globals()[key]).__name__}, got {type(attempt).__name__}"
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
