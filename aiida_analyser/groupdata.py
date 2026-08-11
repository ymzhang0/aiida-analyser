from collections import defaultdict
import inspect
from loguru import logger

def render_process_node_details(node, max_text_chars=120_000):
    """Render a process tree as collapsible HTML for notebook display."""
    import html

    def style_details(margin="4px 0 4px 12px", border="#e0e0e0", background="#f8f9fa"):
        return (
            f"margin: {margin}; border: 1px solid {border}; "
            "border-radius: 4px;"
        ), background

    def read_repository_text(repository, filename):
        try:
            content = repository.get_object_content(filename)
        except Exception as exc:
            return None, f"{exc.__class__.__name__}: {exc}"

        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                return None, "binary or non-UTF-8 content"

        content = str(content)
        if "\x00" in content:
            return None, "binary content"

        truncated = len(content) > max_text_chars
        if truncated:
            content = content[:max_text_chars]
            content += f"\n\n... truncated after {max_text_chars} characters ..."

        return content, None

    def render_remote_path(data, name):
        path = None
        if hasattr(data, "get_remote_path"):
            try:
                path = data.get_remote_path()
            except Exception:
                path = None
        if path is None and hasattr(data, "target_basepath"):
            path = getattr(data, "target_basepath")
        if path is None:
            return None

        computer = getattr(getattr(data, "computer", None), "label", None)
        title = html.escape(name)
        if computer:
            title += f" @ {html.escape(str(computer))}"

        return (
            "<details style='margin: 4px 0 4px 12px; border: 1px solid #d6e4ff; border-radius: 4px;'>"
            f"<summary style='font-weight: 600; cursor: pointer; padding: 6px 10px; background-color: #f0f5ff; border-bottom: 1px solid #d6e4ff;'>{title}</summary>"
            "<div style='padding: 8px 12px; background-color: #ffffff;'>"
            f"<code style='white-space: pre-wrap; word-break: break-all;'>{html.escape(str(path))}</code>"
            "</div></details>"
        )

    def render_repository(data, name):
        if name != "retrieved":
            return None

        repository = getattr(getattr(data, "base", None), "repository", None)
        if repository is None:
            return None

        try:
            names = sorted(repository.list_object_names())
        except Exception as exc:
            return (
                "<span style='color: #c0392b;'>"
                f"Could not list repository objects: {html.escape(str(exc))}"
                "</span>"
            )

        html_str = (
            "<details style='margin: 4px 0 4px 12px; border: 1px solid #dff0d8; border-radius: 4px;'>"
            f"<summary style='font-weight: 600; cursor: pointer; padding: 6px 10px; background-color: #f4fbf1; border-bottom: 1px solid #dff0d8;'>{html.escape(name)} ({len(names)} files)</summary>"
            "<div style='padding: 8px 12px; background-color: #ffffff;'>"
        )

        if not names:
            html_str += "<i style='color: #7f8c8d;'>(empty)</i>"
        for filename in names:
            escaped_filename = html.escape(filename)
            content, error = read_repository_text(repository, filename)
            if content is None:
                html_str += (
                    "<div style='margin-bottom: 6px;'>"
                    f"<code>{escaped_filename}</code>"
                    f" <span style='color: #7f8c8d;'>({html.escape(error or 'not a text file')})</span>"
                    "</div>"
                )
                continue

            html_str += (
                "<details style='margin: 6px 0 6px 12px; border: 1px solid #ecf0f1; border-radius: 4px;'>"
                f"<summary style='cursor: pointer; padding: 5px 8px; background-color: #fbfcfc;'><code>{escaped_filename}</code></summary>"
                "<pre style='margin: 0; padding: 8px; max-height: 360px; overflow: auto; white-space: pre-wrap; word-break: break-word; background-color: #ffffff;'>"
                f"{html.escape(content)}"
                "</pre></details>"
            )

        html_str += "</div></details>"
        return html_str

    def data_to_html(data, name="Parameters"):
        if isinstance(data, dict):
            if not data:
                return "<i style='color: #7f8c8d;'>(empty)</i>"
            style, background = style_details()
            html_str = f"<details style='{style}'>"
            html_str += f"<summary style='font-weight: 600; cursor: pointer; padding: 6px 10px; background-color: {background}; border-bottom: 1px solid #e0e0e0;'>{html.escape(name)} ({len(data)} items)</summary>"
            html_str += "<div style='padding: 8px 12px; background-color: #ffffff;'>"
            for key, value in sorted(data.items(), key=lambda item: str(item[0])):
                key = str(key)
                html_str += f"<div style='margin-bottom: 6px;'><b>{html.escape(key)}:</b> {data_to_html(value, key)}</div>"
            html_str += "</div></details>"
            return html_str

        if isinstance(data, list):
            if not data:
                return "<i style='color: #7f8c8d;'>(empty)</i>"
            style, background = style_details()
            html_str = f"<details style='{style}'>"
            html_str += f"<summary style='font-weight: 600; cursor: pointer; padding: 6px 10px; background-color: {background}; border-bottom: 1px solid #e0e0e0;'>{html.escape(name)} [{len(data)} items]</summary>"
            html_str += "<div style='padding: 8px 12px; background-color: #ffffff;'>"
            for idx, item in enumerate(data):
                html_str += f"<div style='margin-bottom: 6px;'><b>[{idx}]:</b> {data_to_html(item, f'[{idx}]')}</div>"
            html_str += "</div></details>"
            return html_str

        repository_html = render_repository(data, name)
        if repository_html is not None:
            return repository_html

        remote_html = render_remote_path(data, name)
        if remote_html is not None:
            return remote_html

        if hasattr(data, "pk"):
            label = getattr(data, "process_label", data.__class__.__name__)
            return (
                "<span style='background: #eef2f7; padding: 2px 6px; border-radius: 4px; "
                "font-family: monospace; font-size: 0.9em;'>"
                f"Node &lt;{data.pk}&gt; ({html.escape(str(label))})"
                "</span>"
            )

        return f"<span style='font-family: monospace; color: #2c3e50;'>{html.escape(str(data))}</span>"

    def get_process_html(process_node):
        inputs = {}
        for link in process_node.base.links.get_incoming():
            if not link.link_type.value.startswith("call_"):
                val = link.node
                if hasattr(val, "get_dict"):
                    try:
                        inputs[link.link_label] = val.get_dict()
                    except Exception:
                        inputs[link.link_label] = val
                elif hasattr(val, "value"):
                    inputs[link.link_label] = val.value
                else:
                    inputs[link.link_label] = val

        outputs = {}
        for link in process_node.base.links.get_outgoing():
            if link.link_type.value == "return":
                val = link.node
                if hasattr(val, "get_dict"):
                    try:
                        outputs[link.link_label] = val.get_dict()
                    except Exception:
                        outputs[link.link_label] = val
                elif hasattr(val, "value"):
                    outputs[link.link_label] = val.value
                else:
                    outputs[link.link_label] = val

        sub_processes = []
        for link in process_node.base.links.get_outgoing().all():
            if link.link_type.value.startswith("call_"):
                sub_processes.append((link.link_label, link.node))

        process_name = getattr(process_node, "process_label", process_node.__class__.__name__)
        exit_status = getattr(process_node, "exit_status", None)
        exit_status_str = f" (Exit: {exit_status})" if exit_status is not None else ""
        state_emoji = "⏳" if not process_node.is_terminated else ("✅" if process_node.is_finished_ok else "❌")

        html_str = "<details style='margin: 8px 0; border: 1px solid #dcdde1; border-radius: 6px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.03);'>"
        html_str += "<summary style='font-weight: bold; cursor: pointer; padding: 10px 14px; background-color: #f5f6fa; border-bottom: 1px solid #dcdde1;'>"
        html_str += f"<span style='margin-right: 8px;'>{state_emoji}</span>{html.escape(str(process_name))} &lt;{process_node.pk}&gt;{exit_status_str}"
        html_str += "</summary>"
        html_str += "<div style='padding: 12px;'>"

        process_type = getattr(process_node, "process_type", None) or "Unknown"
        process_state = getattr(process_node, "process_state", None)
        process_state_value = getattr(process_state, "value", "Unknown")
        html_str += "<div style='margin-bottom: 8px; font-size: 0.9em; color: #7f8c8d;'>"
        html_str += f"<b>Type:</b> {html.escape(str(process_type))}<br>"
        html_str += f"<b>State:</b> {html.escape(str(process_state_value))}<br>"
        html_str += "</div>"

        html_str += data_to_html(inputs, "Inputs")
        html_str += "<div style='height: 6px;'></div>"
        html_str += data_to_html(outputs, "Outputs")

        if sub_processes:
            html_str += "<div style='height: 10px;'></div>"
            sub_html = "<div style='margin-left: 12px; border-left: 2px dashed #b2bec3; padding-left: 12px;'>"
            sub_html += "<div style='font-size: 0.9em; font-weight: bold; color: #636e72; margin-bottom: 4px;'>Called Processes:</div>"
            for label, sub_node in sorted(sub_processes, key=lambda item: item[1].pk):
                sub_html += f"<div style='margin-bottom: 8px;'><i>Call link: {html.escape(str(label))}</i>"
                sub_html += get_process_html(sub_node)
                sub_html += "</div>"
            sub_html += "</div>"
            html_str += sub_html

        html_str += "</div></details>"
        return html_str

    return get_process_html(node)


class BaseGroupData:
    """Shared helpers for collections of AiiDA process nodes.

    Subclasses can keep a domain-specific nested representation while exposing a
    list of row dictionaries in ``_data``.  The latter automatically gains
    consistent dataframe, filtering, and display behaviour from this class.
    """

    analyser_class = None
    dataframe_columns = ()
    formula_column = 'Material'

    def __init__(self, groups=None):
        self._groups = groups if groups is not None else []
        self._data = defaultdict(lambda: None)  # Subclasses should redefine this
        logger.success(f"Groups imported {self._groups} successfully")

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

    @property
    def available_columns(self):
        """Names available for the public dataframe and table reshaping options.

        Internal ``node`` objects are intentionally excluded: they are kept
        only for filtering and exporting process trees, and are removed before
        the dataframe is returned.
        """
        return self.dataframe_columns

    def iter_group_nodes(self, process_labels=None):
        """Yield nodes from configured groups, optionally filtered by label.

        A missing or inaccessible group is logged and skipped, so one stale
        group label does not prevent the remaining analysis from loading.
        """
        from aiida import orm

        if process_labels is None:
            labels = None
        elif isinstance(process_labels, str):
            labels = {process_labels}
        else:
            labels = set(process_labels)

        for group_label in self._groups:
            try:
                group = orm.load_group(group_label)
            except Exception as exc:
                logger.warning(f'Could not load group {group_label!r}: {exc}')
                continue

            for node in group.nodes:
                if labels is None or getattr(node, 'process_label', None) in labels:
                    yield node

    def _dump_nodes(self, dest, nodes, analyser_class=None, path_builder=None):
        """Copy process trees for *nodes* into a destination directory.

        ``path_builder`` receives a node and returns its path relative to
        ``dest``.  It lets specialised group-data classes keep their semantic
        directory layout while sharing directory creation and error handling.
        """
        from pathlib import Path

        analyser_class = analyser_class or self.analyser_class
        if analyser_class is None:
            raise ValueError(f'{self.__class__.__name__} does not define an analyser_class.')

        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        for node in nodes:
            try:
                relative_path = path_builder(node) if path_builder else Path(str(node.pk))
                analyser_class(node).copy_tree(destination / relative_path)
            except Exception as exc:
                logger.warning(f'Failed to dump node {getattr(node, "pk", "N/A")}: {exc}')

    def _iter_flattened_nodes(self):
        """Yield nodes retained by a list-based flattened data representation."""
        if not isinstance(self._data, list):
            raise NotImplementedError(
                f'{self.__class__.__name__} must provide list-based flattened data or override dump().'
            )

        for row in self._data:
            if 'node' not in row:
                continue
            nodes = row['node']
            if isinstance(nodes, (list, tuple)):
                yield from nodes
            else:
                yield nodes

    def dump(self, dest):
        """Dump the work-chain nodes retained in flattened group data.

        Each flattened row identifies a node analysed by ``analyser_class``.
        Specialised exporters can call :meth:`_dump_nodes` with a custom path
        builder when their directory hierarchy carries domain information.
        """
        self._dump_nodes(dest, self._iter_flattened_nodes())

    @staticmethod
    def get_node_formula(node, default='N/A'):
        """Get a structure formula, falling back to the node extras."""
        try:
            formula = node.inputs.structure.get_formula()
        except Exception:
            formula = default

        if formula != default:
            return formula

        try:
            return node.base.extras.get('formula', default)
        except Exception:
            return default

    @staticmethod
    def _matches_property(analyser, property_filter):
        """Evaluate the compact property-filter syntax used by group tables."""
        if callable(property_filter):
            return bool(property_filter(analyser))
        if not isinstance(property_filter, str):
            return False

        property_name = property_filter.strip()
        negate = property_name.startswith(('not ', '!', '~'))
        if property_name.startswith('not '):
            property_name = property_name[4:].strip()
        elif negate:
            property_name = property_name[1:].strip()
        value = bool(getattr(analyser, property_name, False))
        return not value if negate else value

    def _get_dataframe(self, property_filter=None, values=None):
        """Build a PK-indexed dataframe from flat row dictionaries in ``_data``."""
        import pandas as pd

        rows = self._data if isinstance(self._data, list) else self._flatten_data()
        columns = ['PK', *self.dataframe_columns]
        if not rows:
            return pd.DataFrame(columns=columns).set_index('PK')

        if property_filter:
            if self.analyser_class is None:
                raise ValueError(
                    f'{self.__class__.__name__} does not define analyser_class; '
                    'property_filter is unavailable.'
                )
            filtered_rows = []
            for item in rows:
                try:
                    analyser = self.analyser_class(item['node'])
                    if self._matches_property(analyser, property_filter):
                        filtered_rows.append(item)
                except Exception as exc:
                    logger.warning(f"Error filtering node {item.get('PK', 'N/A')}: {exc}")
            rows = filtered_rows

        dataframe = pd.DataFrame(rows)
        value_keys = self._table_keys(values, 'values')
        if 'node' in dataframe and 'node' not in value_keys:
            dataframe = dataframe.drop(columns='node')
        if 'PK' not in dataframe:
            return dataframe
        return dataframe.set_index('PK').sort_index()

    @staticmethod
    def _filter_by_formula(dataframe, formula_contains=None, formula_match='any', column='Material'):
        """Filter a dataframe by case-insensitive formula substrings."""
        if formula_contains is None or formula_contains == '' or column not in dataframe:
            return dataframe

        terms = [formula_contains] if isinstance(formula_contains, str) else list(formula_contains)
        terms = [str(term).strip().lower() for term in terms if str(term).strip()]
        if not terms:
            return dataframe
        if formula_match not in {'any', 'all'}:
            raise ValueError("formula_match must be either 'any' or 'all'")

        formulas = dataframe[column].astype(str).str.lower()
        masks = [formulas.str.contains(term, regex=False) for term in terms]
        mask = masks[0]
        for next_mask in masks[1:]:
            mask = mask | next_mask if formula_match == 'any' else mask & next_mask
        return dataframe.loc[mask]

    @staticmethod
    def _display_dataframe(dataframe, display_mode, max_height):
        """Display a dataframe in notebook-friendly modes, or return it."""
        mode = 'dataframe' if display_mode is None else str(display_mode).lower()
        if mode in {'dataframe', 'default'}:
            return dataframe
        if mode == 'all':
            import pandas as pd
            from IPython.display import display

            with pd.option_context('display.max_rows', None):
                display(dataframe)
            return None
        if mode == 'scroll':
            from IPython.display import HTML, display

            display(HTML(
                f'<div style="max-height:{int(max_height)}px; overflow:auto;">'
                f'{dataframe.to_html()}</div>'
            ))
            return None
        raise ValueError("display_mode must be one of 'dataframe', 'all', 'scroll', or 'interactive'")

    @staticmethod
    def _table_keys(value, parameter):
        """Normalise one or more dataframe column names passed to ``get_table``."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            keys = list(value)
        except TypeError as exc:
            raise TypeError(f'{parameter} must be a column name or an iterable of column names.') from exc
        if not all(isinstance(key, str) for key in keys):
            raise TypeError(f'{parameter} must contain only column names.')
        return keys

    @classmethod
    def _reshape_table(cls, dataframe, *, index=None, columns=None, values=None, aggfunc='first', fill_value=''):
        """Create an optional hierarchical or pivoted view of a flat dataframe.

        ``index`` accepts one key or a sequence of keys.  A sequence creates a
        pandas ``MultiIndex`` and therefore keeps the requested hierarchy in
        notebook HTML output.  Supplying ``columns`` pivots its key(s) into
        columns; ``values`` identifies the cell value(s).  When the latter is
        omitted, ``Status`` (or ``status``) is selected automatically.
        """
        index_keys = cls._table_keys(index, 'index')
        column_keys = cls._table_keys(columns, 'columns')
        value_keys = cls._table_keys(values, 'values')

        if not index_keys and not column_keys:
            if not value_keys:
                return dataframe
            missing = set(value_keys).difference(dataframe.columns)
            if missing:
                available = ', '.join(map(str, dataframe.columns))
                raise KeyError(f'Unknown table key(s): {sorted(missing)!r}. Available columns: {available}.')
            return dataframe.loc[:, value_keys]

        import pandas as pd

        # A freshly-created dataframe has an anonymous RangeIndex.  Retaining
        # it would make every row a distinct pivot index, so only materialise
        # a meaningful index (notably the PK index of list-based group data).
        if dataframe.index.name is None and isinstance(dataframe.index, pd.RangeIndex):
            flat = dataframe.copy()
        else:
            flat = dataframe.reset_index()
        missing = set(index_keys + column_keys + value_keys).difference(flat.columns)
        if missing:
            available = ', '.join(map(str, flat.columns))
            raise KeyError(f'Unknown table key(s): {sorted(missing)!r}. Available columns: {available}.')

        if not column_keys:
            hierarchical = flat.set_index(index_keys).sort_index()
            return hierarchical.loc[:, value_keys] if value_keys else hierarchical

        if not value_keys:
            for status_key in ('Status', 'status'):
                if status_key in flat.columns:
                    value_keys = [status_key]
                    break
            else:
                raise ValueError(
                    'values is required when pivoting data without a Status or status column.'
                )

        if not index_keys:
            excluded = set(column_keys + value_keys + ['PK', 'node', 'status', 'Status'])
            index_keys = [key for key in flat.columns if key not in excluded]

        pivot = flat.pivot_table(
            index=index_keys,
            columns=column_keys,
            values=value_keys[0] if len(value_keys) == 1 else value_keys,
            aggfunc=aggfunc,
        )
        if fill_value is not None:
            pivot = pivot.fillna(fill_value)
        return pivot.sort_index().sort_index(axis=1)

    @staticmethod
    def get_status_string(node):
        if node is None:
            return 'N/A'

        if not node.is_terminated:
            return '⏳'
        if node.is_finished_ok:
            return '✅'
        if node.is_failed:
            return f'❌ ({node.exit_status})'
        if node.is_excepted:
            return '⚠️ Excepted'
        if node.is_killed:
            return '💀 Killed'
        return f'🏃 {node.process_state.value}'

    def get_table(
        self,
        display_mode='dataframe',
        *,
        max_height=600,
        page_size=25,
        formula_contains=None,
        formula_match='any',
        property_filter=None,
        index=None,
        columns=None,
        values=None,
        aggfunc='first',
        fill_value='',
    ):
        """Return a flat, hierarchical, or pivoted table of group-data rows.

        By default the existing flat PK-indexed table is returned.  ``index``
        can be a key or a sequence of keys to produce a hierarchical
        ``MultiIndex``.  ``columns`` pivots one or more keys into columns and
        ``values`` chooses the cell value(s); it defaults to ``Status`` or
        ``status`` when available.  For example::

            group.get_table(
                index=['Degauss', 'K_Density', 'Q_Density', 'Type'],
                columns='Material',
                values='Status',
            )

        ``aggfunc`` resolves duplicate index/column combinations and defaults
        to ``'first'``.  Set ``fill_value=None`` to retain missing values.
        Without ``columns``, ``values`` selects the returned data columns. For
        example, use ``values='node'`` to return nodes instead of status, or
        ``values=['status', 'node']`` to retain both.  The internal ``node``
        column is otherwise omitted from public dataframes.
        """
        import pandas as pd

        if isinstance(self._data, list):
            dataframe = self._filter_by_formula(
                self._get_dataframe(
                    property_filter=property_filter,
                    values=values,
                ),
                formula_contains=formula_contains,
                formula_match=formula_match,
                column=self.formula_column,
            )
            if str(display_mode).lower() == 'interactive':
                if any(value is not None for value in (index, columns, values)):
                    raise ValueError(
                        'index, columns, and values are not supported in interactive mode.'
                    )
                show_interactive = getattr(self, 'show_interactive', None)
                if show_interactive is None:
                    raise ValueError(f'{self.__class__.__name__} does not provide an interactive table.')
                parameters = inspect.signature(show_interactive).parameters
                kwargs = {}
                if 'max_height' in parameters:
                    kwargs['max_height'] = max_height
                if 'page_size' in parameters:
                    kwargs['page_size'] = page_size
                if 'formula_contains' in parameters:
                    kwargs['formula_contains'] = formula_contains
                if 'formula_match' in parameters:
                    kwargs['formula_match'] = formula_match
                return show_interactive(**kwargs)
            dataframe = self._reshape_table(
                dataframe,
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                fill_value=fill_value,
            )
            return self._display_dataframe(dataframe, display_mode, max_height)

        flattened_list = self._flatten_data()
        if not flattened_list:
            return pd.DataFrame()

        dataframe = pd.DataFrame(flattened_list)
        value_keys = self._table_keys(values, 'values')
        if 'node' in dataframe and 'node' not in value_keys:
            dataframe = dataframe.drop(columns='node', errors='ignore')
        if index is None and columns is None and 'Plane' in dataframe and 'Status' in dataframe:
            # Preserve the historical default for the legacy nested data model.
            columns = 'Plane'
            values = 'Status'

        return self._reshape_table(
            dataframe,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value,
        )

    def _flatten_data(self):
        """To be implemented by subclasses."""
        raise NotImplementedError
