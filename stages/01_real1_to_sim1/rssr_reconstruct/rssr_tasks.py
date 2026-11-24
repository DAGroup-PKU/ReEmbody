"""Explicit backend coverage. An unimplemented task can never inherit di1 gates."""
from rssr_dataset import TASK_IDS

# Expand only with a physical implementation and its task-specific acceptance
# tests. This registry intentionally exposes the current full-dataset gap.
IMPLEMENTED = {task: 'rssr.scene.v2/'+task for task in TASK_IDS}

REQUIREMENTS = {
    'di1': ['select_blue_cube', 'left_holds_container', 'right_places_object', 'released_stable_inside'],
    'di2': ['triangular_prism_geometry', 'select_triangle', 'left_holds_container', 'released_stable_inside'],
    'di3': ['multiple_block_sizes', 'select_largest', 'left_holds_container', 'unselected_objects_preserved'],
    'di4': ['red_cylinder_reference', 'select_initial_left_neighbor', 'left_holds_container', 'released_stable_inside'],
    'di5': ['two_hand_pick', 'exactly_two_in_same_basket', 'third_cube_remains_outside'],
    'di6': ['two_baskets', 'left_cube_in_upper_basket', 'right_cube_in_lower_basket'],
    'di7': ['multiple_geometric_objects', 'all_non_red_in_basket', 'red_cube_remains_outside'],
    'dm1': ['left_holds_container', 'right_places_cube', 'released_stable_inside'],
    'dm2': ['sphere_contact_geometry', 'left_holds_container', 'right_places_ball', 'released_stable_inside'],
    'dm3': ['bottles_geometry', 'all_bottles_upright_on_tray', 'stable_released'],
    'dm4': ['marked_target_region', 'both_cubes_inside_region', 'push_contact_without_grasp'],
    'dm5': ['cup_and_bowl', 'free_granular_particles', 'pour_transfer_fraction', 'no_forced_particle_motion'],
    'dm6': ['articulated_drawer', 'open_before_insert', 'block_inside_drawer', 'closed_after_insert'],
    'dm7': ['two_cube_stack', 'right_cube_first_at_target', 'left_cube_supported_on_right', 'stable_released'],
    'dm8': ['articulated_box_lid', 'open_before_insert', 'block_inside_box', 'closed_after_insert'],
}


def coverage(rows):
    ids = sorted({row['task_id'] for row in rows})
    if set(ids)-set(TASK_IDS):
        raise ValueError('Unrecognized Cobot task family')
    return {'all_supported': all(t in IMPLEMENTED for t in ids),
            'supported': {t: IMPLEMENTED[t] for t in ids if t in IMPLEMENTED},
            'missing': {t: REQUIREMENTS[t] for t in ids if t not in IMPLEMENTED},
            'no_automatic_task_substitution': True}
