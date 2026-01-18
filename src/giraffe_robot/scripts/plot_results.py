import numpy as np
import matplotlib.pyplot as plt
from simulation import run_simulation

final_pos, final_pitch, historical_data = run_simulation()

print('Generating plots...')
j_history = np.array(historical_data['joints'])

fig, axs = plt.subplots(4, 1, figsize=(10, 14))

axs[0].plot(historical_data['time'], historical_data['pos_error'], 'g-', linewidth=2)
axs[0].set_title('Task 1: Position error (m)')
axs[0].set_ylabel('Error [m]')
axs[0].grid(True)

axs[1].plot(historical_data['time'], historical_data['pitch_error'], 'b-', linewidth=2)
axs[1].set_title('Task 2: Pitch error (rad)')
axs[1].set_ylabel('Error [rad]')
axs[1].grid(True)

axs[2].plot(historical_data['time'], historical_data['yaw_error'], 'r-', linewidth=2)
axs[2].set_title('Task 3: Yaw error (rad)')
axs[2].set_ylabel('Error [rad]')
axs[2].grid(True)

labels = ['Pan', 'Tilt', 'Ext', 'W.Pitch', 'W.Roll']
axs[3].set_title('Internal state of joints angles')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Angle (rad) / Length (m)')
axs[3].grid(True)
for j in range(5):
    axs[3].plot(historical_data['time'], j_history[:, j], label=labels[j])
axs[3].legend()  # Legend AFTER plots are added

plt.tight_layout()
plt.show()