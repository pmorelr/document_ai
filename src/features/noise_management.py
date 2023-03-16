# Merge the two classes "Page-Footer" and "Page-Header" into a single class "Noise"

def noise_management(dataset, noise_manag):
    if noise_manag == "all":
        return dataset
    elif noise_manag == "merged":
        dataset = dataset.map(lambda x: {'tags': [0 if y not in [1,4,5] else 1 for y in x['tags']]})
        return dataset
    elif noise_manag == "ignored":
        dataset = dataset.filter(lambda x: "Page-Footer" not in x['tags'] and "Page-Header" not in x['tags'])
        return dataset
    else:
        raise ValueError("Invalid noise management option. Please choose between 'all', 'merged' and 'ignored'")