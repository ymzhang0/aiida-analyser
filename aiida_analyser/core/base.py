from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
from .workchains import clean_workdir
from aiida.tools import delete_nodes
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from collections import deque
import itertools

from .logging import get_console, get_logger


logger = get_logger(__name__)
console = get_console()
_COPY_TREE_DEPTH: ContextVar[int] = ContextVar('aiida_analyser_copy_tree_depth', default=0)


def _format_node_ref(node: orm.Node) -> str:
    """Return a compact process reference for log messages."""
    process_label = getattr(node, 'process_label', node.__class__.__name__)
    node_pk = getattr(node, 'pk', 'N/A')
    return f'{process_label}<{node_pk}>'


def _summarize_child_labels(labels: list[str], max_items: int = 6) -> str:
    """Return a compact summary of child labels for log messages."""
    if not labels:
        return 'no direct children'

    if len(labels) <= max_items:
        return ', '.join(labels)

    head = ', '.join(labels[:max_items])
    return f'{head}, ... (+{len(labels) - max_items} more)'


@contextmanager
def _copy_tree_logging_scope(
    node: orm.Node,
    destpath: Path,
    child_labels: list[str] | None = None,
):
    """Emit a compact top-level extraction log while suppressing nested noise."""
    depth = _COPY_TREE_DEPTH.get()
    token = _COPY_TREE_DEPTH.set(depth + 1)
    resolved_destpath = destpath.resolve()
    node_ref = _format_node_ref(node)

    if child_labels is None:
        detail = None
    else:
        detail = f'{len(child_labels)} direct children: {_summarize_child_labels(child_labels)}'

    try:
        if depth == 0:
            message = f'[bold cyan]extract[/] {node_ref} -> [blue]{resolved_destpath}[/]'
            if detail:
                message += f' [dim]({detail})[/]'
            logger.info(message)
        yield
        if depth == 0:
            logger.info(f'[green]complete[/] {node_ref}')
    except Exception:
        logger.exception(f'[red]failed[/] extracting {node_ref} -> {resolved_destpath}')
        raise
    finally:
        _COPY_TREE_DEPTH.reset(token)


@dataclass
class ProcessTree:
    """
    A tree structure to represent the processes of a workchain.
    """
    name: str = 'ROOT'  # The name of the node (e.g. 'pw_relax', 'iteration_01')
    node: Optional[orm.WorkChainNode | orm.CalcJobNode] = None  # The AiiDA node object (WorkChainNode or CalcJobNode)
    children: Dict[str, 'ProcessTree'] = field(default_factory=dict) # The children nodes, indexed by the name
    
    # Overload the constructor to build the tree from the original dictionary
    def __init__(self, aiida_node: orm.WorkChainNode | orm.CalcJobNode, name: str = 'ROOT'):
        """
        Initialize the ProcessTree node and recursively build the child tree.
        
        :param aiida_node: The AiiDA node object (WorkChainNode or CalcJobNode).
        :param name: The name of the current node (for the root node can be any string, for the child nodes is the link_label).
        """
        self.name = name
        self.node = aiida_node
        self.children = {}
        
        # Only WorkChainNode has 'called' subprocesses
        # We use try-except block to handle CalcJobNode or other nodes without .called attribute
        try:
            # Iterate over all subprocesses called by the current node
            subprocesses = list(aiida_node.called)
            subprocesses.sort(key=lambda p: p.ctime)
            
            for subprocess in subprocesses:
                
                # Extract the link_label of the subprocess, as the name of the child node
                # Assume all subprocesses have metadata_inputs and contain call_link_label
                try:
                    link_label = subprocess.base.attributes.all['metadata_inputs']['metadata']['call_link_label']
                except Exception:
                    # If no label, use the pk or uuid of the subprocess as a fallback
                    link_label = subprocess.base.attributes.all.get('process_label', f"unlabeled_process_{subprocess.pk}")

                # Recursively create the ProcessTree child node
                # The power of this is that it can handle CalcJobNode stopping the recursion,
                # and WorkChainNode continuing the recursion.
                
                # Key point: Directly call ProcessTree(subprocess, link_label)
                # This will delegate the recursive construction logic to the ProcessTree constructor of the child node
                child_node = ProcessTree(aiida_node=subprocess, name=link_label)
                
                # Add the child node to the children dictionary of the current node
                self.children[link_label] = child_node

        except AttributeError:
            # If the node does not have the .called attribute (e.g. CalcJobNode or other non-WorkChainNode),
            # an AttributeError will be raised, and we stop the recursion, the children dictionary remains empty.
            pass

    # Core: Implement __getitem__ magic method
    def __getitem__(self, key: str) -> 'ProcessTree':
        """
        Allow the use of square brackets [] syntax to access the child nodes.
        For example: root_tree['pw_relax']
        """
        if key in self.children:
            return self.children[key]
        else:
            # if key does not exist, raise KeyError, to mimic the dictionary behavior
            raise KeyError(f"Child node with name '{key}' not found in ProcessTree.")

    def __getattr__(self, name: str) -> 'ProcessTree':
        """
        Allow the use of dot . syntax to access the child nodes.
        For example: root_tree.pw_relax
        
        Note: This method is only called when the object does not find the attribute named 'name'.
        """
        # Check if this name exists in the child nodes dictionary
        if name in self.children:
            return self.children[name]
        else:
            # if name does not exist in children, and is not a property of ProcessTree itself,
            # raise AttributeError, to mimic the standard object behavior
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute or child named '{name}'")

    # Optional: Implement __contains__ method to support 'key in tree' check
    def __contains__(self, key: str) -> bool:
        """
        Allow the use of 'in' keyword to check if the child node exists.
        For example: 'pw_relax' in root_tree
        """
        return key in self.children

    def find_last_node(self):
        """
        Use BFS to find the bottom right node of the process tree.
        """
        queue = deque([self])
        last_node = None
        while queue:
            current_node = queue.popleft()
            last_node = current_node
            if current_node.children:
                for child_node in current_node.children.values():
                    queue.append(child_node)
        return last_node

    def print(self):
        """
        Print the process tree.
        """
        console.print(self.name)
        for child in self.children.values():
            child.print()

    def print_tree(self, prefix: str = "", is_last: bool = True):
        """
        Manually print the tree structure to the console.
        """
        
        # Determine the prefix and connector line of the current node
        connector = "└── " if is_last else "├── "
        
        # Get the node information
        node_id = getattr(self.node, 'pk', 'N/A')
        node_type = self.node.process_label
        label = f"{self.name} ({node_type} PK: {node_id})"
        
        # Print the current node
        console.print(prefix + connector + label)
        
        # Determine the indentation of the next layer
        # If the current node is not the last child node, the next layer needs to continue using the vertical line '│ '
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # Recursively print the child nodes
        children_list = list(self.children.values())
        for i, child in enumerate(children_list):
            is_last_child = (i == len(children_list) - 1)
            child.print_tree(prefix=next_prefix, is_last=is_last_child)

    def print_nodes_info(
            self,
            target_node_type: str, 
            extractor: Callable[[Any], Dict[str, Any]],
            prefix: str = "",
            is_last: bool = True,
        ) -> None:
        """
        Recursively traverse the ProcessTree, collect the information of all matching target type nodes.

        :param target_node_type: The target AiiDA node type string (e.g. 'WorkChainNode').
        :param extractor: A function that takes an AiiDA node and returns a dictionary containing the desired information.
        :return: A list of dictionaries containing the information of all matching nodes.
        """
        
        connector = "└── " if is_last else "├── "
        
        # Get the node information
        node_id = getattr(self.node, 'pk', 'N/A')
        node_type = self.node.node_type
        process_label = self.node.process_label
        label = f"{self.name} ({process_label} PK: {node_id})"
        next_prefix = prefix + ("    " if is_last else "│   ")                        
        # 1. Check if the current node matches the target node type
        if target_node_type == node_type:
            # If matched, use the provided extractor function to extract the information
            info = extractor(self.node)
            console.print(prefix + connector + label + ": " + info)
        else:
            console.print(prefix + connector + label)
        # 2. Recursively traverse the child nodes
        children_list = list(self.children.values())
        for i, child in enumerate(children_list):
            is_last_child = (i == len(children_list) - 1)
            child.print_nodes_info(target_node_type, extractor, next_prefix, is_last_child)

    @staticmethod
    def traverse_and_check(
        node: 'ProcessTree',
        current_path: str,
        ) -> tuple[str, 'ProcessTree'] | None: # Explicitly specify the return type
        """
        Traverse the ProcessTree and check if the node is the first errored CalcJobNode.
        
        :returns: (path_to_errored_node, ProcessTree_node) or None
        """
        # 1. Construct the full path of the current node
        new_path = f"{current_path}/{node.name}" if current_path else node.name
        
        # 2. Check if the current node is the target type and errored
        if node.node.node_type == 'process.calculation.calcjob.CalcJobNode.' and not node.node.is_finished_ok:
            # Base Case 1: Found the errored node, return the result immediately
            # This is also a case where the recursion chain stops
            return (new_path, node)
        
        # 3. Recursively traverse the child nodes
        for child_node in node.children.values():
            result = ProcessTree.traverse_and_check(node=child_node, current_path=new_path)
            
            # If the child call found the result, pass the result up the call chain
            if result is not None:
                return result
                
        # 4. Explicit Base Case 2: Traversed the current branch, no errored node found
        # Must explicitly return None, to indicate the upper level of the call chain: This branch is safe
        return None

    @staticmethod
    def _copy_tree(node: 'ProcessTree', destpath: Path) -> None:
        """
        Recursively traverse the ProcessTree, find the CalcJobNode and extract its input files to the local directory.

        :param node: The current ProcessTree node.
        :param current_path: The corresponding directory of the current node in the local file system.
        """
        
        # 1. Create the directory of the current node
        # Use the name of the node as the directory name (e.g. 'pw_relax', 'iteration_01')
        node_dir = destpath / node.name

        # 2. Check if the current node is a CalcJobNode
        if node.node.node_type == 'process.calculation.calcjob.CalcJobNode.':
            # Copy the input files of the CalcJobNode to the destination directory
            node_dir.mkdir(parents=True, exist_ok=True)
            
            calcjob_node = node.node
            calcjob_node.base.repository.copy_tree(node_dir)
            calcjob_node.outputs.retrieved.copy_tree(node_dir)

        # 3. Recursively process the child nodes
        for child_node in node.children.values():
            try:
                ProcessTree._copy_tree(child_node, node_dir)
            except Exception as e:
                logger.warning(
                    'Failed to copy child subtree %s under %s: %s',
                    child_node.name,
                    node.name,
                    e,
                )

    def copy_tree(self, destpath: Path) -> Path:
        """
        Extract the input files of all CalcJobNodes from the entire ProcessTree and save them to the local directory.

        :param root_directory_name: The name of the root directory in the local file system.
        :return: The Path object of the created root directory in the local file system.
        """
        
        with _copy_tree_logging_scope(self.node, destpath, list(self.children.keys())):
            for child_node in self.children.values():
                self._copy_tree(child_node, destpath)
        return destpath

class WorkChainAnalyser(ABC):
    """
    BaseAnalyser for the WorkChain.
    """

    def __init__(self, workchain: orm.WorkChainNode):
        self.node = workchain
    @abstractmethod
    def get_source(self):
        """Get the source of the workchain."""
        pass
    @abstractmethod
    def clean_workchain(self, dry_run=True):    
        """Clean the workchain."""
        pass


class BaseCalculationAnalyser:
    """Base analyser for a CalcJob node."""

    def __init__(self, calculation: orm.CalcJobNode):
        self.node = calculation

    @property
    def node_ref(self) -> str:
        """Return a compact process reference for the current node."""
        return _format_node_ref(self.node)

    def get_state(self):
        """Return the state of the calculation node."""
        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0

        return (
            'ROOT',
            self.node.process_state.value,
            self.node.exit_code if self.node.is_finished else None,
        )

    def copy_tree(self, destpath: Path) -> Path:
        """Copy the input repository and retrieved outputs of a calculation."""
        with _copy_tree_logging_scope(self.node, destpath):
            destpath.mkdir(parents=True, exist_ok=True)
            self.node.base.repository.copy_tree(destpath)

            try:
                self.node.outputs.retrieved.copy_tree(destpath)
            except (AttributeError, KeyError):
                pass

        return destpath

    def get_calcjob_paths(self) -> dict[str, str]:
        """Return the remote path of this calculation using a root-relative key."""
        if not self.node.is_finished_ok:
            return {}

        try:
            remote_path = self.node.outputs.remote_folder.get_remote_path()
        except (AttributeError, KeyError):
            return {}

        return {'ROOT': remote_path}


class BaseWorkChainAnalyser(WorkChainAnalyser):
    """
    BaseAnalyser for the WorkChain.
    """
    _RY2eV    = 13.605693122990
    _RYA22Jm2 = 4.3597447222071E-18/2 * 1E+20
    _eVA22Jm2 = 1.602176634E-19 * 1E+20

    @property
    def node_ref(self) -> str:
        """Return a compact process reference for the current workchain."""
        return _format_node_ref(self.node)

    @staticmethod
    def _format_status_markup(process_state: str, exit_code: Any) -> str:
        """Return a Rich-marked process state string."""
        normalized_exit_code = getattr(exit_code, 'status', exit_code)
        if process_state == 'finished_ok' and normalized_exit_code == 0:
            return f'[green]{process_state}[/]'
        if normalized_exit_code not in (None, 0):
            return f'[red]{process_state}[/]'
        return f'[yellow]{process_state}[/]'

    def _log_source_missing(self) -> None:
        """Emit a uniform warning when source metadata is missing."""
        logger.warning(f'[yellow]missing source[/] {self.node_ref}')

    def _log_state_summary(self, path: str, process_state: str, exit_code: Any) -> None:
        """Emit a compact state summary for the current workchain."""
        normalized_exit_code = getattr(exit_code, 'status', exit_code)
        status_markup = self._format_status_markup(process_state, normalized_exit_code)
        message = (
            f'{self.node_ref} status={status_markup} '
            f'path=[magenta]{path}[/] exit_code=[bold]{normalized_exit_code}[/]'
        )
        if process_state == 'finished_ok' and normalized_exit_code == 0:
            logger.info(message)
        else:
            logger.warning(message)

    def _print_text_block(self, title: str, body: str) -> None:
        """Print a titled multiline text block using the shared console."""
        console.rule(f'[bold]{title}[/]')
        console.print(body)

    @staticmethod
    def split_source(source: str | tuple[str, str] | None) -> tuple[str, str] | None:
        """Normalize supported source representations to a ``(db, id)`` tuple."""
        if source is None:
            return None
        if isinstance(source, tuple):
            return source
        if '-' not in source:
            raise ValueError(f'Invalid source format: {source!r}')
        return tuple(source.split('-', 1))

    def _get_node_from_tree(self, label: str) -> orm.Node:
        """
        Helper method to get a node from the process tree by its label.
        Raises AttributeError if the label is not found.
        """
        if label not in self.process_tree:
            raise AttributeError(f"'{label}' is not found in the process tree of WorkChain<{self.node.pk}>")
        return self.process_tree[label].node

    def _get_child_labels(
        self,
        labels: tuple[str, ...] = (),
        prefixes: tuple[str, ...] = (),
        process_label: str | None = None,
    ) -> list[str]:
        """Return direct child labels filtered by exact names, prefixes, and process label."""
        matches = []
        seen = set()

        for label in labels:
            if label not in self.process_tree:
                continue
            child_tree = self.process_tree[label]
            if process_label and child_tree.node.process_label != process_label:
                continue
            matches.append(label)
            seen.add(label)

        for child_name, child_tree in self.process_tree.children.items():
            if child_name in seen:
                continue
            if prefixes and not any(child_name.startswith(prefix) for prefix in prefixes):
                continue
            if process_label and child_tree.node.process_label != process_label:
                continue
            if labels or prefixes or process_label:
                matches.append(child_name)
                seen.add(child_name)

        return matches

    @staticmethod
    def _get_safe_energy(node: orm.Node) -> float | None:
        """
        Safely retrieve the 'energy' from output_parameters.
        """
        if not node:
            return None
        if 'output_parameters' not in node.outputs._get_keys():
            return None
        return node.outputs.output_parameters.get('energy')

    @staticmethod
    def _get_calcjob_paths(processes_tree, parent_label=''):
        """
        Recursively extract all CalcJob remote paths from the nested dictionary created by get_processes_dict.

        :param processes_dict: The dictionary generated by get_processes_dict.
        :param parent_label: The parent path for building hierarchical labels (used internally for recursion).
        :return: A flattened dictionary { 'full label': 'remote path' }.
        """
        flat_paths = {}
        for name, node in processes_tree.children.items():

            full_label = f"{parent_label}/{name}" if parent_label else name

            if not node.children:
                if node.node.is_finished_ok and node.node.node_type == 'process.calculation.calcjob.CalcJobNode.':
                    remote_path = node.node.outputs.remote_folder.get_remote_path()
                    flat_paths[full_label] = remote_path
            else:
                nested_paths = BaseWorkChainAnalyser._get_calcjob_paths(
                    node,
                    parent_label=full_label
                )
                flat_paths.update(nested_paths)

        return flat_paths

    def get_calcjob_paths(self):
        """Get the remote paths of the all CalcJobNodes in the process tree."""
        return self._get_calcjob_paths(self.process_tree)

    @staticmethod
    def _join_relative_path(child_name: str, relative_path: str) -> str:
        """Join a direct child label with a child analyser's relative path."""
        if relative_path in ('', 'ROOT'):
            return child_name
        return f'{child_name}/{relative_path}'

    @cached_property
    def process_tree(self):
        """Get the ProcessTree of the workchain."""
        return ProcessTree(self.node)

    def _get_state_from_subprocesses(self, subprocesses, required_subprocesses=()):
        """
        Resolve the first unfinished subprocess in execution order.

        :param subprocesses: Iterable of ``(link_label, analyser_class)`` pairs.
        :param required_subprocesses: Namespaces that must exist in the process tree.
        """
        process_tree = self.process_tree
        required = set(required_subprocesses)

        for subprocess_name, subprocess_analyser in subprocesses:
            if subprocess_name not in process_tree:
                if subprocess_name in required:
                    return self._get_state_from_tree()
                continue

            subprocess_node = process_tree[subprocess_name].node
            if subprocess_node.is_finished_ok:
                continue

            analyser = subprocess_analyser(subprocess_node)
            path, process_state, exit_code = analyser.get_state()
            return (
                subprocess_name if path == 'ROOT' else f'{subprocess_name}/{path}',
                process_state,
                exit_code,
            )

        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0

        return self._get_state_from_tree()

    def print_process_tree(self):
        """Print the process tree."""
        self.process_tree.print_tree()

    @staticmethod
    def get_retrieved(node):
        """Get the retrieved of the all workchains."""
        retrieved = {}

        for subprocess in node.called:
            if 'CalcJobNode' in subprocess.node_type:
                link_label = subprocess.base.attributes.all['metadata_inputs']['metadata']['call_link_label']
                retrieved[link_label] = subprocess.outputs.retrieved if subprocess.outputs.retrieved else None

            elif 'WorkChainNode' in subprocess.node_type:
                link_label = subprocess.base.attributes.all['metadata_inputs']['metadata']['call_link_label']
                retrieved[link_label] = {}
                sub_paths = BaseWorkChainAnalyser.get_retrieved(subprocess)
                retrieved[link_label].update(sub_paths)
            else:
                pass
        return retrieved

    def copy_tree(
        self,
        destpath: Path,
        ):
        """Copy the tree of the workchain to the destination directory."""
        self.process_tree.copy_tree(destpath)

    def _copy_tree_for_direct_children(
        self,
        destpath: Path,
        analyser_resolver: Callable[[str, ProcessTree], Any],
    ) -> Path:
        """
        Copy the tree by delegating each direct child to its own analyser.

        If ``analyser_resolver`` returns ``None`` for a child, fall back to the
        legacy tree copy for that direct subtree.
        """
        with _copy_tree_logging_scope(self.node, destpath, list(self.process_tree.children.keys())):
            destpath.mkdir(parents=True, exist_ok=True)

            for child_name, child_tree in self.process_tree.children.items():
                analyser_class = analyser_resolver(child_name, child_tree)

                if analyser_class is None:
                    ProcessTree._copy_tree(child_tree, destpath)
                    continue

                analyser = analyser_class(child_tree.node)
                analyser.copy_tree(destpath / child_name)
        return destpath

    def _get_calcjob_paths_for_direct_children(
        self,
        analyser_resolver: Callable[[str, ProcessTree], Any],
    ) -> dict[str, str]:
        """
        Collect calcjob remote paths by delegating each direct child to its own analyser.

        If ``analyser_resolver`` returns ``None`` for a child, fall back to the
        legacy recursive ProcessTree traversal for that subtree.
        """
        flat_paths = {}

        for child_name, child_tree in self.process_tree.children.items():
            analyser_class = analyser_resolver(child_name, child_tree)

            if analyser_class is None:
                child_paths = self._get_calcjob_paths(child_tree)
            else:
                child_paths = analyser_class(child_tree.node).get_calcjob_paths()

            for relative_path, remote_path in child_paths.items():
                full_label = self._join_relative_path(child_name, relative_path)
                flat_paths[full_label] = remote_path

        return flat_paths
    
    def _get_state_from_tree(self):
        """
        Helper method to get state by traversing the process tree.
        This is a utility method for subclasses to use in their get_state() implementation.
        
        Returns:
            tuple: (path, exit_status, message) or None if no error found
        """
        if self.node.is_finished_ok:
            return 'ROOT', 'finished_ok', 0

        last_node = self.process_tree.find_last_node()
        if last_node is not None and not last_node.node.is_finished_ok:
            return (
                last_node.name,
                last_node.node.process_state.value,
                last_node.node.exit_code if last_node.node.is_finished else None,
            )
            
        return 'ROOT', 'unknown_status', None

    def print_state(self, print_output=False, print_stdout=False, print_stderr=False):
        """
        Print the state of the workchain.
        This method requires get_state() to be implemented in subclasses.
        """
        try:
            path, process_state, exit_code = self.get_state()
        except AttributeError:
            logger.warning(f'{self.node_ref} get_state() is not implemented.')
            return -1

        normalized_exit_code = getattr(exit_code, 'status', exit_code)

        self._log_state_summary(path, process_state, normalized_exit_code)

        # If exit_code is an integer and non-zero, try to get detailed output
        if isinstance(normalized_exit_code, int) and normalized_exit_code != 0:
            result = ProcessTree.traverse_and_check(node=self.process_tree, current_path='')
            if result:
                path, node = result
                if print_output:
                    try:
                        if 'aiida.out' in node.node.get_retrieve_list():
                            self._print_text_block(
                                'Standard Output',
                                node.node.get_retrieved_node().get_object_content('aiida.out'),
                            )
                    except (AttributeError, KeyError):
                        logger.warning(f'{self.node_ref} aiida.out not found in retrieved.')
                if print_stdout:
                    try:
                        if '_scheduler-stdout.txt' in node.node.get_retrieve_list():
                            self._print_text_block(
                                'Scheduler Stdout',
                                node.node.get_scheduler_stdout(),
                            )
                    except (AttributeError, KeyError):
                        logger.warning(f'{self.node_ref} scheduler stdout not found in retrieved.')
                if print_stderr:
                    try:
                        if '_scheduler-stderr.txt' in node.node.get_retrieve_list():
                            self._print_text_block(
                                'Scheduler Stderr',
                                node.node.get_scheduler_stderr(),
                            )
                    except (AttributeError, KeyError):
                        logger.warning(f'{self.node_ref} scheduler stderr not found in retrieved.')
                return normalized_exit_code

        if process_state == 'finished_ok':
            return 0

        return normalized_exit_code if isinstance(normalized_exit_code, int) else -1
    
    def get_source(self):
        """Get the source of the workchain."""
        try:
            source_db, source_id = self.node.base.extras.get_many(('source_db', 'source_id'))
        except Exception:
            return None
        return f"{source_db}-{source_id}"

    def clean_workchain(self, dry_run=True):
        """Clean the workchain."""

        cleaned_calcs = clean_workdir(self.node, dry_run=dry_run)
        message = f'Cleaned the workchain <{self.node.pk}>:\n'
        message += '  ' + ' '.join(map(str, cleaned_calcs)) + '\n'
        message += f'Deleted the workchain <{self.node.pk}>:\n'
        deleted_nodes, _ = delete_nodes([self.node.pk], dry_run=dry_run)
        message += '  ' + ' '.join(map(str, deleted_nodes))

        return message, True

    @staticmethod
    def extract_remote_path(node: orm.CalcJobNode) -> str:
        """
        Extract the remote path of the node.
        """
        return f"remote path: {node.outputs.remote_folder.get_remote_path()}"

    def print_remote_paths(self):
        """
        Print the remote paths of the all CalcJobNodes in the process tree.
        """
        self.process_tree.print_nodes_info(target_node_type='process.calculation.calcjob.CalcJobNode.', extractor=self.extract_remote_path)

    @staticmethod
    def extract_retrieved(node: orm.CalcJobNode) -> str:
        """
        Extract the remote path of the node.
        """
        return f"retrieved: {node.outputs.retrieved.uuid}"

    def print_retrieved(self):
        """
        Print the retrieved of the all CalcJobNodes in the process tree.
        """
        self.process_tree.print_nodes_info(target_node_type='process.calculation.calcjob.CalcJobNode.', extractor=self.extract_retrieved)
