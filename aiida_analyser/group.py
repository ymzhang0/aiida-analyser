from aiida import load_profile
from aiida.orm import QueryBuilder, Group, Node
from copy import deepcopy
from collections.abc import Mapping

def recursive_merge(d1: dict, d2: dict) -> dict:
    """
    Recursively merge two dictionaries.

    Args:
        d1 (dict): The first dictionary.
        d2 (dict): The second dictionary.

    Returns:
        dict: The recursively merged dictionary.
    """
    
    # 1. 从 d1 开始，创建一个深拷贝 (deepcopy)
    merged = deepcopy(d1)
    
    # 2. 遍历 d2 的所有键值对
    for key, value_d2 in d2.items():
        
        # 3. 检查这个键是否已经存在于 'merged' (来自 d1) 中
        if key in merged:
            value_merged = merged[key]
            
            # 4. Start checking merge logic
            
            # 4.1. Both are mappings -> recursively merge
            if isinstance(value_merged, Mapping) and isinstance(value_d2, Mapping):
                merged[key] = recursive_merge(value_merged, value_d2)
            
            # 4.2. Both are lists -> concatenate
            elif any(
                (isinstance(value_merged, t) and isinstance(value_d2, t))
                for t in [list, set, tuple]
                ):
                merged[key] = value_merged + value_d2
            
            # 4.5. All other cases -> d2 overrides d1
            else:
                merged[key] = deepcopy(value_d2)
        else:
            # 5. Key only exists in d2 -> directly add
            merged[key] = deepcopy(value_d2)
            
    return merged

def set_description_for_group(group, description, overwrite=False):
    if type(group) == str:
        group = Group.collection.get(label=group)

    print("old description: ", group.description)
    if overwrite:
        group.description = description
    else:
        group.description += description
    print("new description: ", group.description)

def count_groups(profile, log=print):
    load_profile(profile, allow_switch=True)
    qb = QueryBuilder().append(
        Group,
        project=["*"]
    )
    log(f"{'pk':<10} {'label':<35} {'count':<10}")
    for group in qb.all(flat=True):
        log(f"{group.pk:<10} {group.label:<35} {group.count():<10}")

def count_nodes(profile, node_type, process_type):
    load_profile(profile, allow_switch=True)
    qb = QueryBuilder().append(Node, filters={
        "node_type": node_type,
        "process_type": process_type
    })
    return qb.count()

def get_and_count_types(profile, log=print):
    load_profile(profile, allow_switch=True)
    qb = QueryBuilder().append(Node, project=['node_type', 'process_type']).distinct()
    log(f"{'count':<10} {'node_type':<50} {'process_type':<10}")
    for [node_type, process_type] in qb.all():
        if process_type:
            count = count_nodes(node_type, process_type)
            log(f"{count:<10} {node_type:<60} {process_type:<60}")
        else:
            count = count_nodes(node_type, None)
            log(f"{count:<10} {node_type:<60} {'None':<60}")

def delete_groups(LABEL_PATTERN, PERFORM_DELETION, DELETE_NODES = False):
    from aiida.tools import delete_nodes

    """Finds and deletes the groups."""
    qb = QueryBuilder()
    qb.append(Group, filters={'label': {'like': LABEL_PATTERN}})

    groups_to_delete = qb.all(flat=True)

    if not groups_to_delete:
        print(f"✅ No groups found matching the pattern '{LABEL_PATTERN}'. Nothing to do.")
        return

    print(f"Found {len(groups_to_delete)} groups to delete:")
    for group in groups_to_delete:
        print(f"  - PK: {group.pk}, Label: {group.label}")

    if PERFORM_DELETION:
        pks_to_delete = [g.pk for g in groups_to_delete]
        if DELETE_NODES:
            for g in groups_to_delete:
                delete_nodes(list(node.pk for node in g.nodes), dry_run=False)

        print("\n" + "="*20)
        print("PERFORMING DELETION...")
        for pk in pks_to_delete:
            Group.collection.delete(pk)
        print(f"✅ Successfully deleted {len(pks_to_delete)} groups.")
        print("="*20)
    else:
        print("\n" + "="*20)
        print("DRY RUN MODE: No groups were deleted.")
        print("To delete these groups, edit the script and set PERFORM_DELETION = True")
        print("="*20)

def get_sourced_nodes(group_label, profile=None):
    if profile:
        load_profile(profile, allow_switch=True)
    qb = QueryBuilder().append(
        Group, filters={'label': group_label}, tag='group'
        ).append(
        Node, project=['*'], with_group='group', tag='nodes'
        )
    sourced_nodes = {}
    for node in qb.all(flat=True):
        try:
            source_db = node.base.extras.all['source_db']
            source_id = node.base.extras.all['source_id']
            source = f"{source_db}-{source_id}"
            sourced_nodes[source] = node
        except:
            pass
    return sourced_nodes

def get_sourced_nodes_all_groups(profile):
    load_profile(profile, allow_switch=True)
    qb = QueryBuilder().append(Group, project=['*'])
    sourced_nodes = {}
    for group in qb.all(flat=True):
        sourced_nodes[group.label] = get_sourced_nodes(group.label)
    return sourced_nodes

def get_group_status(group_label, analyser_class, profile = None):
    if profile:
        load_profile(profile, allow_switch=True)
    
    results = {}

    qb = QueryBuilder().append(Group, filters={'label': group_label}, tag='group'
        ).append(
        Node, project=['*'], with_group='group', tag='nodes'
        )
    for node in qb.all(flat=True):
        analyser = analyser_class(node)
        path, status, message = analyser.get_state()
        results = recursive_merge(results, {
            path: {
                status: {
                    'message': message,
                    'nodes': [node.uuid]
                }
            }
        })

    return results