import colorama as col

import runtime_arguments as rarg

if rarg.args.problem == 'square_a':
    # fixed height of the PI, slip (n.v = 0) on the PI
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_a'
elif rarg.args.problem == 'square_b':
    # fixed slope of the PI (grad z = t r_hat, free height), no slip on the PI: the setup of Ferraro & Castellana, PRE
    rmsh = 'mesh.read.square'
    vp = 'variational_problem_bc_square_b'
else:
    raise ValueError(f"problem {rarg.args.problem} not implemented for the stability analysis with flows")

print(f'{col.Fore.CYAN}Loaded {rarg.args.problem} problem (stability, flow){col.Style.RESET_ALL}')
