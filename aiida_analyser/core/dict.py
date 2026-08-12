"""Dictionary helpers used throughout the package."""

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

    def __repr__(self):
        return f"NestedDict({self.data})"

    def to_dict(self):
        """Recursively convert NestedDict to native dict."""
        result = {}
        for key, value in self.data.items():
            if isinstance(value, NestedDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
