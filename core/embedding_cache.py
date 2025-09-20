from collections import OrderedDict

class EmbeddingCache(OrderedDict):
    def __init__(self, max_items=10):
        super().__init__()
        self.max_items = max_items

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_items:
            self.popitem(last=False)  # remove oldest
