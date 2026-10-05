import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_no_circle':
    rmsh = 'read_mesh_square_no_circle'
    vp = 'variational_problem_bc_square_no_circle'
    vp_pp = 'variational_problem_pp_square_no_circle'
    prout_bc = 'print_out_bc_square_no_circle'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square_no_circle'



elif rarg.args.problem == 'square':
    rmsh = 'read_mesh_square'
    vp = 'variational_problem_bc_square'
    vp_pp = 'variational_problem_pp_square'
    prout_bc = 'print_out_bc_square'
    prout_forces_on_boundaries = 'print_out_force_on_boundaries_bc_square'

elif rarg.args.problem in ('cylinder', 'channel'):
    # 'cylinder': unbounded flow past a cylinder, 'channel': Schaefer-Turek (DFG) channel with a cylinder
    rmsh = 'read_mesh_cylinder'
    vp = f'variational_problem_bc_{rarg.args.problem}_steady'

# steady variational problem (residual F, BCs, mass form) used by the stability analysis
vp_steady = f'variational_problem_bc_{rarg.args.problem}_steady'

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem{col.Style.RESET_ALL}')