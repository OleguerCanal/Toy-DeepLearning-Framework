
def dict_to_string(dictionary):
    s = str(dictionary)
    s = s.replace(" ", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("'", "")
    s = s.replace(":", "-")
    s = s.replace(",", "_")
    return s

def g(a, b, c):
    print(a)
    print(b)
    print(c)

def f(a, **kwargs):
    print(a)
    g(a, **kwargs)
    s = dict_to_string(kwargs)
    print(s)

if __name__ == "__main__":
    param_space = {
        "a": 1,
        "b": 2,
        "c": 3
    }
    f(**param_space)

