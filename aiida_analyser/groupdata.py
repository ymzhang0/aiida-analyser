from collections import defaultdict


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
    """Base class for group data objects."""

    def __init__(self, groups=None):
        self._groups = groups if groups is not None else []
        self._data = defaultdict(lambda: None)  # Subclasses should redefine this

    @property
    def groups(self):
        return self._groups

    @property
    def data(self):
        return self._data

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

    def get_table(self):
        """Generic table generation with pivoting."""
        import pandas as pd

        flattened_list = self._flatten_data()
        if not flattened_list:
            return pd.DataFrame()

        df = pd.DataFrame(flattened_list)

        # 1. Define the columns we know are NOT part of the index
        # 'Plane' is our pivot column, 'Status' is our value
        pivot_col = 'Plane'
        value_col = 'Status'

        if pivot_col in df.columns and value_col in df.columns:
            index_cols = [col for col in df.columns if col not in [pivot_col, value_col]]
            pivot_df = df.pivot_table(
                values=value_col,
                index=index_cols,
                columns=pivot_col,
                aggfunc='first',
            )
            pivot_df = pivot_df.fillna('')
            pivot_df = pivot_df.sort_index(axis=1)
            return pivot_df

        return df

    def _flatten_data(self):
        """To be implemented by subclasses."""
        raise NotImplementedError
