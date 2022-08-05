import Evolution as ev

import shutil
import os
import time

# Define the parameters

deltat = 1e-3
lenght = 200
p_max = 0.5
p_newspacing = 0
steps = int(1e2)

alphas = 0.1
Qs = 0.1
f0 = 0.01
init_params = [alphas, Qs, f0]

Nc = 3
Nf = 3
params = [Nc]

save = 10
plot = 10

# Check if data directory exists and ask for remove
if os.path.isdir('data'):
    inp = input("Some data are already saved. Type 'y' to remove it.\n")
    if inp == 'y':
        shutil.rmtree('data')
    else:
        exit()


# Execute the simulation

start = time.time()

evol = ev.Evolution(deltat, lenght, p_max, p_newspacing, init_params, params,
                    save=save, plot=plot)

evol.evolve(steps)

print(time.time() - start)


# Shutdown the computer
#time.sleep(60)
#os.system('shutdown now')
