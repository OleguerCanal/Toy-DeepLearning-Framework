
def dict_to_string(dictionary):
    s = str(dictionary)
    s = s.replace(" ", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("'", "")
    s = s.replace(":", "-")
    s = s.replace(",", "_")
    return s