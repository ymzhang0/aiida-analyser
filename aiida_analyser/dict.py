from collections import UserDict

class NestedDict(UserDict):
    """A dictionary that automatically creates new NestedDict instances for nested access."""
    def __getitem__(self, key):
            try:
                return self.data[key]
            except KeyError:
                new_nested_dict = NestedDict()
                self.data[key] = new_nested_dict
                return new_nested_dict
            
    # You can add other dictionary methods, such as __repr__ to make the print output better
    def __repr__(self):
        return f"NestedDict({self.data})"

    def to_dict(self):
            """
            Recursively convert NestedDict to native dict.
            """
            result = {}
            for key, value in self.data.items():
                if isinstance(value, NestedDict):
                    inner_dict = value.to_dict()
                    result[key] = inner_dict
                else:
                    result[key] = value
            return result