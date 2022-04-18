

class Label:

    name = None
    code = None

    def __init__(self, name, code):
        assert(type(name) is str)
        assert(type(code) is int)

        self.name = name
        self.code = code

        


