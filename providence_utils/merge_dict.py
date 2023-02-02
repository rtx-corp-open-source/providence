"""
Module: merge_dict.py (renamed from nested_dict.py)
Description:
    (VENDORED from the *actual* providence-tools)
    A small module of two functions, meant for nesting and unnesting of dictionaries.
    To facilitae some configuration-based iteration of programs, I wanted to iterate
    configurations articulated as dictionaries while also finding a way to nest configurations (as only some
    of the top level parameters need regular changing, but lower-level items *do*).

    I see these functions - as they stand - as keys to facilitating that.

Author: Stephen Fox
Date: 2021/08/27

Author's Note:
----------------

The following is undefined behavior
nest_values({
    "a.b": 12,
    "a.b.c"; 11
})

... mainly because, depending on the version of Python 3 you're using, the dict type is
backed by the legacy hash-table implementation, an OrderedTreeMap, or something else (future).

Please, just supply flatten, non-conflicting keys to `nest_values()`.
If you supply a key collision in the dict you're providing to `nest_keys()` Python will either
  1. yell at you, for supplying the constructor of `dict` with a name collision
  2. last-in-wins on the curlies-and-str-literal dictionary. (Check this)


This implementation currently doesn't support copying keys, mostly because it would slow down this
code at minimal value. If you're using immutability as a discipline / design-pattern, this code
will treat you just right. If you want to implement the deep copy in a way that doesn't suck 
(so far, the fastest I've seen if to have a separate code path, but that doubles the code size)
you're more than welcome to PR this.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from typing import Mapping


def merge_dictionaries(d1: Mapping, d2: Mapping) -> dict:
    """
    Merge nested dictionaries, made for merging dictionaries; whenever collision occurs,
    favors the Mapping value rather than anything else. If neither value is a mapping, apply an `or`
    comparison and let Python figure it out the "truthier" result
    """
    left_keys = d1.keys() - d2.keys()
    right_keys = d2.keys() - d1.keys()
    common_keys = d1.keys() & d2.keys()

    output = dict()
    for key in common_keys:
        left, right = d1[key], d2[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            new_value = merge_dictionaries(left, right)
            output[key] = new_value
        elif isinstance(left, Mapping):
            output[key] = left
        elif isinstance(right, Mapping):
            output[key] = right
        else:
            output[key] = left or right  # what else can we do.

    for key in left_keys: output[key] = d1[key]
    for key in right_keys: output[key] = d2[key]

    return output

