def intersection(set1:set, set2:set) -> set:
    return {element for element in set1 if element in set2}

def is_subset(set1:set, set2:set) -> bool:
    "Returns true if set1 is a subset of set2"
    return intersection(set1, set2) == set1
