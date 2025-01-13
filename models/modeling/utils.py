
def get_attributes_info(name):
    if "CUB" in name:
        info = {
            "input_dim" : 312,
            "n" : 200,
            "m" : 50,
            "g":28
        }
    elif "AwA" in name:
        info = {
            "input_dim": 85,
            "n": 50,
            "m": 10,
            "g":9
        }
    elif "SUN" in name:
        info = {
            "input_dim": 102,
            "n": 717,
            "m": 72,
            "g":4
        }
    else:
        info = {}
    return info

