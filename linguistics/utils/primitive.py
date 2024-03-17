import datetime
from typing import Iterable


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def ngrams(items: list, n: int):
    assert 0 < n, 'incorrect `n` value.'
    results = []
    for start in range(len(items) - n + 1):
        end = start + n
        n_gram = items[start:end]
        if n_gram:
            results.append(n_gram)
    return results


def subgroups(items: Iterable, by: [int, str], starts_from: [int, str] = 0, scope: list = None):
    """
    split single list to partitioned list by attribute of elements
    :param items: a single iterable object
    :param by: if you want to split the list by particular attribute (ex. key of dict) then pass str,
               or you can pass int value (ex. specific index of every individual)
    :param starts_from: initial value to be compared with `by`
    :param scope: if you pass a sequence as a `scope`, then it will return only expected outputs
                  corresponding to the attributes or index of individuals in the range of scope.
                  (ex. ['key1', 'key2', ...] or [2, 3, 4, ...])
    :return: a list of lists
    """
    result = list()
    buffer = list()
    current_state = starts_from
    if scope and starts_from not in scope:
        raise ValueError(f'Wrong arguments - starts_from: {starts_from} ({type(starts_from)}), scope: {scope}')
    for element in items:
        if isinstance(by, str) and isinstance(element, dict):
            criterion = element[by]
        elif isinstance(by, str) and isinstance(element, tuple):  # namedtuple
            criterion = getattr(element, by)
        elif isinstance(by, int) and isinstance(element, (list, tuple)):
            criterion = element[by]
        elif isinstance(by, str):
            criterion = element.__getattribute__(by)
        else:
            raise ValueError(f'Wrong arguments - item: {element} ({type(element)}), by: {by} ({type(by)})')
        if scope and criterion not in scope:
            continue
        if current_state == criterion:
            buffer.append(element)
        else:
            current_state = criterion
            result.append(buffer)
            buffer = list()
            buffer.append(element)
    result.append(buffer)  # last pang
    return result
