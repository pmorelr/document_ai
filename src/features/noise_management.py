# Merge the two classes "Page-Footer" and "Page-Header" into a single class "Noise"

def noise_management(dataset, noise_manag):
    if noise_manag == "all":
        return dataset
    elif noise_manag == "binary":
        dataset = dataset.map(lambda x: {'tags': [0 if y not in [1,4,5] else 1 for y in x['tags']]})
        return dataset
    elif noise_manag == "triplet":
        reclass = [0,1,0,0,2,3]
        dataset = dataset.map(lambda x: {'tags': [0 if y not in [1,4,5] else reclass[y] for y in x['tags']]})
        return dataset
    else:
        raise ValueError("Invalid noise management option. Please choose between 'all', 'binary' and 'triplet'")