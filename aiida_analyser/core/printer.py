"""Utilities for printing nested data structures."""

from collections.abc import Iterable, Mapping
from html import escape


class Printer:
    """Render nested data as a text tree or a collapsible notebook tree."""

    prefix_item = '├── '
    prefix_last_item = '└── '
    prefix_indent = '    '
    prefix_parent = '│   '

    _print = print

    def __init__(self, data: dict | list):
        """Create a printer for a mapping or iterable.

        ``aiida.orm.Dict`` values are recursively converted to regular Python
        dictionaries so that their complete content is rendered instead of
        their abbreviated node representation.
        """
        normalised = self._normalise(data)
        if not self._is_collection(normalised):
            raise TypeError('Input data must be a map or iterable.')

        self.data = normalised

    def print(self):
        """Display a collapsible tree in notebooks and a text tree elsewhere."""
        if self._in_notebook():
            from IPython.display import display

            display(self)
            return

        if not self.data:
            self._print(self.data)
            return
        self._print_recursive(self.data)

    def _repr_html_(self):
        """Return a collapsible HTML representation for Jupyter frontends."""
        if not self.data:
            return f'<pre class="aiida-analyser-printer-empty">{escape(repr(self.data))}</pre>'

        content = self._html_items(self.data)
        return f'''<div class="aiida-analyser-printer">
<style>
.aiida-analyser-printer {{
  font-family: var(--jp-code-font-family, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace);
  font-size: var(--jp-code-font-size, 13px);
  line-height: 1.55;
}}
.aiida-analyser-printer ul {{
  border-left: 1px solid #9aa0a655;
  list-style: none;
  margin: 0 0 0 .55em;
  padding-left: 1.25em;
}}
.aiida-analyser-printer > ul {{ border-left: 0; margin-left: 0; padding-left: 0; }}
.aiida-analyser-printer li {{ margin: .12em 0; }}
.aiida-analyser-printer summary {{ cursor: pointer; width: fit-content; }}
.aiida-analyser-printer summary:hover .aa-printer-key {{ text-decoration: underline; }}
.aiida-analyser-printer .aa-printer-key {{ color: var(--jp-mirror-editor-variable-color, #795e26); }}
.aiida-analyser-printer .aa-printer-value {{
  color: var(--jp-mirror-editor-string-color, inherit);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.aiida-analyser-printer .aa-printer-meta {{
  color: var(--jp-ui-font-color2, #666);
  font-family: var(--jp-ui-font-family, sans-serif);
  font-size: .9em;
  margin-left: .65em;
}}
</style>
<ul>{content}</ul>
</div>'''

    def _print_recursive(self, data, prefix: str = ''):
        items = self._items(data)

        for index, (key, value) in enumerate(items):
            is_last = index == len(items) - 1
            connector = self.prefix_last_item if is_last else self.prefix_item
            self._print(f'{prefix}{connector}{key}')

            new_prefix = prefix + (self.prefix_indent if is_last else self.prefix_parent)
            if self._is_collection(value):
                self._print_recursive(value, new_prefix)
            else:
                self._print(f'{new_prefix}{self.prefix_last_item}{self._format_value(value)}')

    def _html_items(self, data, depth: int = 0):
        rows = []
        for key, value in self._items(data):
            safe_key = escape(str(key))
            if self._is_collection(value):
                count = len(value) if hasattr(value, '__len__') else None
                item_label = 'item' if count == 1 else 'items'
                count_label = f'{count} {item_label}' if count is not None else 'items'
                metadata = f'{type(value).__name__} · {count_label}'
                opened = ' open' if depth == 0 else ''
                children = self._html_items(value, depth + 1)
                rows.append(
                    f'<li><details{opened}><summary><span class="aa-printer-key">{safe_key}</span>'
                    f'<span class="aa-printer-meta">{escape(metadata)}</span></summary>'
                    f'<ul>{children}</ul></details></li>'
                )
            else:
                rows.append(
                    f'<li><span class="aa-printer-key">{safe_key}</span>: '
                    f'<span class="aa-printer-value">{escape(self._format_value(value))}</span></li>'
                )
        return ''.join(rows)

    @classmethod
    def _normalise(cls, value):
        if cls._is_orm_scalar(value):
            return value.value

        if cls._is_orm_dict(value):
            value = value.get_dict()

        if isinstance(value, Mapping):
            return {key: cls._normalise(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._normalise(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._normalise(item) for item in value)
        if cls._is_collection(value):
            return [cls._normalise(item) for item in value]
        return value

    @classmethod
    def _format_value(cls, value):
        if cls._is_installed_code(value):
            computer = getattr(value, 'computer', None)
            computer_label = getattr(computer, 'label', None)
            label = getattr(value, 'label', None)
            pk = getattr(value, 'pk', None)
            if label is not None and computer_label is not None:
                return f'Remote code {label!r} on {computer_label}, pk: {pk}'

        return repr(value)

    @classmethod
    def _is_orm_dict(cls, value):
        """Identify ``orm.Dict`` without importing AiiDA solely for rendering."""
        return callable(getattr(value, 'get_dict', None)) and cls._has_aiida_base(value, 'Dict')

    @classmethod
    def _is_orm_scalar(cls, value):
        return hasattr(value, 'value') and any(
            cls._has_aiida_base(value, class_name) for class_name in ('Int', 'Float', 'Bool')
        )

    @classmethod
    def _is_installed_code(cls, value):
        return cls._has_aiida_base(value, 'InstalledCode')

    @staticmethod
    def _has_aiida_base(value, class_name):
        return any(
            base.__name__ == class_name and base.__module__.startswith('aiida.') for base in type(value).__mro__
        )

    @staticmethod
    def _is_collection(value):
        return isinstance(value, Mapping) or (
            isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray))
        )

    @staticmethod
    def _items(data):
        if isinstance(data, Mapping):
            return list(data.items())
        return [(f'[{index}]', value) for index, value in enumerate(data)]

    @staticmethod
    def _in_notebook():
        try:
            from IPython import get_ipython
        except ImportError:
            return False

        shell = get_ipython()
        return shell is not None and getattr(shell, 'kernel', None) is not None
