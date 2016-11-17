config = {
    'path_fea_dir': '../feature',
    'path_fea_meta': '../feature/.meta',
    'path_fea_impl': '../fea_impl.py',
    'path_data_dir': '../data/',
    'path_data_meta': '../data/.meta',
    'path_model_dir': '../model/',
    'path_model_meta': '../model/.meta',
    'path_task_meta': '../.meta',
    'backend': 'numpy',
    'dtype': 'float32',
    'mean': 0.0,
    'stddev': 0.01,
    'maxval': 0.01,
    'minval': -0.01,
    'seed': 0,
}

nodes = {}
links = {}


def str_2_value(str_value):
    try:
        return int(str_value)
    except ValueError:
        return float(str_value)


def general_max(x):
    if check_type(x, 'list') or check_type(x, 'set'):
        if len(x) > 0:
            return max(x)
    else:
        return x


def general_len(x):
    if check_type(x, 'list') or check_type(x, 'set'):
        return len(x)
    else:
        return 1


def as_array(x):
    if check_type(x, 'list'):
        return x
    else:
        return [x]


def check_type(data, dtype):
    dtype_name = type(data).__name__.lower()
    if dtype == 'int':
        return 'int' in dtype_name
    elif dtype == 'float':
        return 'float' in dtype_name
    elif dtype == 'str':
        return 'str' in dtype_name
    elif dtype == 'list':
        return 'list' in dtype_name or 'array' in dtype_name or 'tuple' in dtype_name
    elif dtype == 'set':
        return 'set' in dtype_name
    elif dtype == 'dict':
        return 'map' in dtype_name or 'dict' in dtype_name
    elif dtype == 'agg':
        return 'set' in dtype_name or 'tuple' in dtype_name or \
               'list' in dtype_name or 'array' in dtype_name or \
               'dict' in dtype_name
    elif dtype == 'none':
        return data == 'None' or data == 'none' or data is None
    elif dtype == 'data':
        return 'data' in dtype_name


def is_int(data):
    return check_type(data, 'int')


def is_float(data):
    return check_type(data, 'float')


def is_val(data):
    return is_int(data) or is_float(data)


def is_str(data):
    return check_type(data, 'str')


def is_list(data):
    return check_type(data, 'list')


def is_set(data):
    return check_type(data, 'set')


def is_dict(data):
    return check_type(data, 'dict')


def is_none(data):
    return check_type(data, 'none')


def is_agg(data):
    return check_type(data, 'agg')


def is_data(data):
    return check_type(data, 'data')
