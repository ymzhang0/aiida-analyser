from unittest import result
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from enum import Enum
from collections import OrderedDict
from abc import ABC, abstractmethod
from pathlib import Path
from .workchains import clean_workdir
from aiida.tools import delete_nodes
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from collections import deque


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
        print(self.name)
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
        print(prefix + connector + label)
        
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
            print(prefix + connector + label + ": " + info)
        else:
            print(prefix + connector + label)
        # 2. Recursively traverse the child nodes
        children_list = list(self.children.values())
        for i, child in enumerate(children_list):
            is_last_child = (i == len(children_list) - 1)
            child.print_nodes_info(target_node_type, extractor, next_prefix, is_last_child)

    @staticmethod
    def traverse_and_check(
        node: 'ProcessTree',
        current_path: str,
        ) -> tuple[str, str] | None: # Explicitly specify the return type
        """
        Traverse the ProcessTree and check if the node is the first errored CalcJobNode.
        
        :returns: (path_to_errored_node, process_state) or None
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
            ProcessTree._copy_tree(child_node, node_dir)

    def copy_tree(self, destpath: Path) -> Path:
        """
        Extract the input files of all CalcJobNodes from the entire ProcessTree and save them to the local directory.

        :param root_directory_name: The name of the root directory in the local file system.
        :return: The Path object of the created root directory in the local file system.
        """
        
        print(f"Starting extraction to directory: {destpath.resolve()}")
        
        # Start the recursion. From the child nodes of the root node, and use root_path as the parent directory for these child nodes.
        for child_node in self.children.values():
            self._copy_tree(child_node, destpath)
            
        print("Extraction complete.")
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

class BaseWorkChainAnalyser(WorkChainAnalyser):
    """
    BaseAnalyser for the WorkChain.
    """

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

    @property
    def process_tree(self):
        """Get the ProcessTree of the workchain."""
        return ProcessTree(self.node)

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
        
    def get_state(self):
        """Get the state of the workchain."""
        if self.node.is_finished_ok:
            return 'ROOT', 0, 'finished OK'    
        else:
            result = ProcessTree.traverse_and_check(node=self.process_tree, current_path='')
            if not result:
                return 'ROOT', -1, 'Unknown status'
            else:
                path, node = result
                exit_code = node.node.exit_code
                return path, exit_code.status, exit_code.message  
    
    def print_state(self, print_output=False, print_stdout=False, print_stderr=False):
        """Print the state of the workchain."""
        if self.node.is_finished_ok:
            print(f"WorkChain<{self.node.pk}> is finished OK.")
            return 0
        result = ProcessTree.traverse_and_check(node=self.process_tree, current_path='')
        if not result:
            print(f"Can't check the state of WorkChain<{self.node.pk}>.")
            return -1
        path, node = result
        exit_code = node.node.exit_code

        print(
            f"WorkChain<{self.node.pk}> exit with {exit_code.status} at {path}.\n"
            f"    Message: {exit_code.message}"
        )

        if exit_code.status:
            if print_output:
                if 'aiida.out' not in node.node.get_retrieve_list():
                    print('aiida.out not found in retrieved')
                    return
                print('Found in standard output:')
                print(node.node.get_retrieved_node().get_object_content('aiida.out'))
            if print_stdout:
                if '_scheduler-stdout.txt' not in node.node.get_retrieve_list():
                    print('_scheduler-stdout.txt not found in retrieved')
                    return
                print('Found in standard output:')
                print(node.node.get_scheduler_stdout())
            if print_stderr:
                if '_scheduler-stderr.txt' not in node.node.get_retrieve_list():
                    print('_scheduler-stderr.txt not found in retrieved')
                    return
                print('Found in standard error:')
                print(node.node.get_scheduler_stderr())
            return exit_code.status
        else:
            return -1
    
    def get_source(self):
        """Get the source of the workchain."""
        try:
            source_db, source_id = self.node.base.extras.get_many(('source_db', 'source_id'))
        except Exception:
            return None
        return f"{source_db}-{source_id}"

    def clean_workchain(self, exempted_states, dry_run=True):
        """Clean the workchain."""

        path, status, message   = self.get_state()
        message = f'Process<{self.node.pk}> is now {status} at {path}. Please check if you really want to clean this workchain.\n'
        if status in exempted_states:
            print(message)
            return message, False
        cleaned_calcs = clean_workdir(self.node, dry_run=dry_run)
        message += f'Cleaned the workchain <{self.node.pk}>:\n'
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