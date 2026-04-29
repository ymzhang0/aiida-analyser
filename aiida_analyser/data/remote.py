from aiida.engine import calcfunction
from aiida.orm import RemoteStashFolderData, Str, Computer
from pathlib import Path

@calcfunction
def move_stashed_folder(
    source_stash_folder_data: RemoteStashFolderData,
    target_computer_label: Str,
    target_remote_path: Str,
) -> RemoteStashFolderData:
    """Create a new RemoteStashFolderData node pointing to an already existing folder
    on another computer.

    Warning:
        This function does NOT copy files. It only creates the AiiDA data node.
    """

    source_path = Path(source_stash_folder_data.target_basepath)
    source_list = source_stash_folder_data.source_list
    stash_mode = source_stash_folder_data.stash_mode
    suffix = Path(*source_path.parts[-3:])

    target_path = Path(target_remote_path.value) / suffix

    computer = Computer.collection.get(label=target_computer_label.value)

    target_stash = RemoteStashFolderData(
        computer=computer,
        stash_mode=stash_mode,
        target_basepath=str(target_path),
        source_list=source_list,
    )

    return target_stash