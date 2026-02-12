from typing import Any
import math
import numpy as np

def round_to_significant_digits(
        x : float, 
        digits : int
    ) -> float:
    """
    Rounds a given number to the specified number of significant digits.

    Parameters:
        x (float): The number to be rounded.
        digits (int): The number of significant digits to round to.

    Returns:
        float: The number rounded to the specified significant digits. 
                If the input number is 0, it is returned as is.

    Example:
        round_to_significant_digits(1234.5678, 3) -> 1230.0
        round_to_significant_digits(0.001234, 2) -> 0.0012
    """
    if x == 0:
        return float(x)
    else:
        return float(round(x, -int(math.floor(math.log10(abs(x)))) + (digits - 1)))

def get_values_recursively(
        dic : dict, 
        key : str, 
        startswith : bool
    ) -> list:
    """
    Recursively retrieves values from a nested dictionary where the keys match a given key or start with a given prefix.

    Args:
        dic (dict): The dictionary to search through.
        key (str): The key to match or use as a prefix.
        startswith (bool): If True, matches keys that start with the given key; otherwise, matches keys that are exactly equal.

    Returns:
        list: A list of values whose keys match the criteria.

    Note:
        This function searches recursively through all nested dictionaries.
    """
    found = []
    for k, v in dic.items():
        if k == key or (startswith and key.startswith(k)):
            found.append(v)
        elif isinstance(v, dict):
            found.extend(get_values_recursively(v, key=key, startswith=startswith))
    return found

def get_one_value_recursively(
        dic : dict, 
        key : str, 
        startswith : bool
    ) -> Any:
    """
    Recursively retrieves a single value from a nested dictionary based on a key.

    Args:
        dic (dict): The dictionary to search through.
        key (str): The key to look for in the dictionary.
        startswith (bool): If True, matches keys that start with the given key; otherwise, matches exact keys.

    Returns:
        Any: The single value found for the specified key.

    Raises:
        ValueError: If zero or more than one value is found for the specified key.
    """
    found_values = get_values_recursively(dic, key, startswith)
    if not len(found_values) == 1:
        raise ValueError(f'looking in {dic=} for {key=} found {found_values=}')
    return found_values[0]

def set_values_recursively(
        dic : dict, 
        key : str, 
        value : Any,
        startswith : bool
    ) -> int:
    """
    Recursively sets values in a nested dictionary where the keys match a given key or start with a given prefix.

    Args:
        dic (dict): The dictionary to search through.
        key (str): The key to match or use as a prefix.
        value (any): The value to set.
        startswith (bool): If True, matches keys that start with the given key; otherwise, matches keys that are exactly equal.

    Returns:
        int: Number of times the values has been set.

    Note:
        This function searches recursively through all nested dictionaries.
    """
    found = 0
    for k, v in dic.items():
        if k == key or (startswith and key.startswith(k)):
            dic[k] = value
            found += 1
        elif isinstance(v, dict):
            found += set_values_recursively(v, key=key, value=value, startswith=startswith)
    return found

def set_one_value_recursively(
        dic : dict, 
        key : str, 
        value : Any,
        startswith : bool
    ) -> None:
    """
    Recursively sets a single value in a nested dictionary based on a key.

    Args:
        dic (dict): The dictionary to search through.
        key (str): The key to look for in the dictionary.
        startswith (bool): If True, matches keys that start with the given key; otherwise, matches exact keys.

    Returns:
        Any: The single value found for the specified key.

    Raises:
        ValueError: If zero or more than one matching key is found.
    """
    found_values = set_values_recursively(dic, key, value, startswith)
    if not found_values == 1:
        raise ValueError(f'looking in {dic=} for {key=} found {found_values=}')

def strip_suffix(
        s : str, 
        require_strict_colon : bool = False
    ) -> str:
    """
    Removes trailing digits from the end of a string.

    Args:
        s (str): The input string.

    Returns:
        str: The string with trailing digits removed.
    """
    if require_strict_colon:
        if ':' in s:
            ind = len(s) -1
            while ind > 0:
                if s[ind] == ':':
                    break
                else: ind -= 1
            # only remove number!
            if s[ind+1:].isdigit():
                s = s[:ind]
    else:
        while len(s) > 0:
            if s[-1] ==':' or s[-1].isdigit():
                s = s[:-1]
            else:
                break
    return s

def merge_dicts(dict1 : dict, dict2 : dict) -> dict:
    """
    Merges two dictionaries into a single dictionary, handling duplicate keys.
    If both dictionaries contain the same key (that is not a string), a KeyError is raised.
    For string keys present in both dictionaries, their values are collected into a list.
    If a key has only one value, it is added as-is to the merged dictionary.
    If a key has multiple values, each value is added with a modified key in the format 'key:idx'.
    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge.
    Returns:
        dict: The merged dictionary with unique keys and indexed duplicates.
    Raises:
        KeyError: If there are duplicate non-string keys in both dictionaries.
    """
    if any(k in dict2.keys() for k in dict1.keys() if not isinstance(k, str)):
        raise KeyError(f'Duplicate keys which are not str in dict1 and dict2: {[k for k in dict1.keys() if (not isinstance(k, str) and k in dict2.keys())]}')
    dict_lists : dict[str,list] = {}
    dict_lists = add_dict(dict_lists, dict1)
    dict_lists = add_dict(dict_lists, dict2)
    
    dict_merged = {}
    for k, l in dict_lists.items():
        if len(l) == 1:
            dict_merged[k] = l[0]
        else:
            dict_merged.update({f'{k}:{idx}' : v for idx, v in enumerate(l)})
    return dict_merged

def add_dict(dict_complete : dict, dict_add : dict) -> dict[Any, Any]:
    """
    Adds key-value pairs from `dict_add` to `dict_complete`, grouping values by key.

    If a key in `dict_add` is not a string and does not exist in `dict_complete`, it is added with its value in a list.
    If a key is a string, its suffix is stripped using `strip_suffix`, and the value is appended to a list under the processed key in `dict_complete`.

    Args:
        dict_complete (dict): The dictionary to which items will be added.
        dict_add (dict): The dictionary containing items to add.

    Returns:
        dict[Any, Any]: The updated dictionary with added and grouped values.

    Raises:
        AssertionError: If a non-string key from `dict_add` already exists in `dict_complete`.
    """
    for k, v in dict_add.items():
        if not isinstance(k, str):
            assert k not in dict_complete, f'{k=}, {dict_complete.keys()=}, {dict_add.keys()=}'
            dict_complete[k] = [v]
            continue
        key = strip_suffix(k, True)
        dict_complete.setdefault(key, []).append(v)
    return dict_complete

def get_key_startswith(
        dic : dict[Any, Any], 
        key : str
    ) -> Any | None:
    """
    Returns the value from a dictionary whose key starts with the specified string.

    Args:
        dic (dict[Any, Any]): The dictionary to search.
        key (str): The prefix string to match at the start of the keys.

    Returns:
        Any: The value corresponding to the single key that starts with the given prefix,
             or None if there is not exactly one such key.
    """
    possible_values = [v for k, v in dic.items() if isinstance(k, str) and k.startswith(key)]
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        raise KeyError(f'None of the dict keys {list(dic.keys())=} starts with {key=}.')

def get_key_startswith_all(
        dic : dict[Any, Any], 
        key : str
    ) -> list:
    """
    Returns a list of values from the dictionary whose string keys start with the specified prefix.

    Args:
        dic (dict[Any, Any]): The dictionary to search.
        key (str): The prefix to match at the start of each key.

    Returns:
        list: A list of values whose keys are strings starting with the given prefix.
    """
    return [v for k, v in dic.items() if isinstance(k, str) and k.startswith(key)]

def get_startswith_key(
        dic : dict[Any, Any], 
        key : str
    ) -> Any | None:
    """
    Returns the value from a dictionary whose key matches the start of the specified string.

    Args:
        dic (dict[Any, Any]): The dictionary to search.
        key (str): The prefix string to contain the start of the key.

    Returns:
        Any: The value corresponding to the single key that is the prefix of the given string,
             or None if there is not exactly one such key.
    """
    possible_values = [(k, v) for k, v in dic.items() if isinstance(k, str) and key.startswith(k)]
    if len(possible_values) == 1:
        return possible_values[0][1]
    elif len(possible_values) > 1:
        possible_values = [(k, v) for (k, v) in possible_values if all(k.startswith(l) for (l, _) in possible_values)]
        if len(possible_values) != 1:
            raise KeyError(f'{key=} starts with multiple of {list(dic.keys())=}?')
        else:
            return possible_values[0][1]
    raise KeyError(f'{key=} doesnt start with one of {list(dic.keys())=}')


### conversion between Watts, dBm, gray
def W_dBm(x : np.ndarray) -> np.ndarray:
    """
    Converts power values from Watts to decibel-milliwatts (dBm).

    Parameters
    ----------
    x : np.ndarray
        Array of power values in Watts.

    Returns
    -------
    np.ndarray
        Array of power values in dBm.

    Notes
    -----
    Ignores divide-by-zero warnings; returns -inf for zero input values.
    """
    with np.errstate(divide='ignore'):
        return 10 * np.log10(x) + 30
    
def dBm_W(x : np.ndarray) -> np.ndarray:
    """
    Converts power values from decibel-milliwatts (dBm) to watts (W).

    Parameters
    ----------
    x : np.ndarray
        Input array containing power values in dBm.

    Returns
    -------
    np.ndarray
        Array of power values converted to watts (W).
    """
    return np.power(10, (x - 30)/10)


def gray_dBm(x : np.ndarray, pl_trnc : float, pl_max : float) -> np.ndarray:
    """
    Applies a linear transformation to an input array to scale its values between a truncated path loss and a maximum path loss in dBm.

    Parameters:
        x (np.ndarray): Input array of normalized values (typically in [0, 1]).
        pl_trnc (float): Truncated path loss value in dBm (lower bound).
        pl_max (float): Maximum path loss value in dBm (upper bound).

    Returns:
        np.ndarray: Array of scaled values in dBm.
    """
    return x * (pl_max - pl_trnc) + pl_trnc

def dBm_gray(x : np.ndarray, pl_trnc : float, pl_max : float, clip : bool) -> np.ndarray:
    """
    Normalizes dBm values to a grayscale range [0, 1].

    Parameters:
        x (np.ndarray): Input array of dBm values.
        pl_trnc (float): Truncation value; values below this are mapped to 0.
        pl_max (float): Maximum value; values above this are mapped to 1.
        clip (bool): If True, output values are clipped to [0, 1]. If False, values may fall outside this range.

    Returns:
        np.ndarray: Array of normalized grayscale values.
    """
    y = (x - pl_trnc) / (pl_max - pl_trnc) 
    return np.clip(y, 0, 1) if clip else y