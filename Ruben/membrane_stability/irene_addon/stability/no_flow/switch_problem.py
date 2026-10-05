import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'ring':
    rmsh = 'mesh.read.ring'
    vp = 'variational_problem_bc_ring'
elif rarg.args.problem == 'square_a':
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_a'
else:
    raise ValueError(f"problem {rarg.args.problem} not implemented for the stability analysis")

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem (stability){col.Style.RESET_ALL}')
