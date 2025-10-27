import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# =========================
# PARAMETER TANGGA
# =========================
STEP_LENGTH = 40
STEP_HEIGHT = 20
N_STEPS = 8
WIDTH = 120
L_TOTAL = STEP_LENGTH * N_STEPS

# =========================
# FUNGSI PERMUKAAN TANGGA
# =========================
def plot_stairs(ax):
    for i in range(N_STEPS):
        x0, x1 = i * STEP_LENGTH, (i + 1) * STEP_LENGTH
        z0, z1 = -i * STEP_HEIGHT, -(i + 1) * STEP_HEIGHT

        # permukaan horizontal
        X, Y = np.meshgrid([x0, x1], [-WIDTH/2, WIDTH/2])
        Z = np.full_like(X, z0)
        ax.plot_surface(X, Y, Z, color='lightgray', edgecolor='k', alpha=0.7)

        # permukaan vertikal
        X, Y = np.meshgrid([x1, x1], [-WIDTH/2, WIDTH/2])
        Z = np.array([[z0, z1], [z0, z1]])
        ax.plot_surface(X, Y, Z, color='gray', edgecolor='k', alpha=0.7)

def z_stair(x):
    i = np.floor(x / STEP_LENGTH)
    i = np.clip(i, 0, N_STEPS - 1)
    return -i * STEP_HEIGHT

# =========================
# MEMBUAT KUBUS 3D
# =========================
def make_cube(xc, yc, zc, size=30, rot_yaw=0):
    d = size/2
    v = np.array([
        [-d, -d, -d],
        [ d, -d, -d],
        [ d,  d, -d],
        [-d,  d, -d],
        [-d, -d,  d],
        [ d, -d,  d],
        [ d,  d,  d],
        [-d,  d,  d],
    ])

    # rotasi yaw di sekitar sumbu Z
    Rz = np.array([
        [np.cos(rot_yaw), -np.sin(rot_yaw), 0],
        [np.sin(rot_yaw),  np.cos(rot_yaw), 0],
        [0, 0, 1]
    ])
    v_rot = v @ Rz.T + np.array([xc, yc, zc])

    faces = [
        [v_rot[j] for j in [0,1,2,3]],
        [v_rot[j] for j in [4,5,6,7]],
        [v_rot[j] for j in [0,1,5,4]],
        [v_rot[j] for j in [2,3,7,6]],
        [v_rot[j] for j in [1,2,6,5]],
        [v_rot[j] for j in [4,7,3,0]]
    ]
    return faces

# =========================
# ANIMASI ROBOT LABA-LABA
# =========================
def animasi_robot_spider_halo():
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_stairs(ax)

    # Badan dan kepala
    body = Poly3DCollection([], facecolors='skyblue', edgecolors='k', alpha=0.9)
    head = Poly3DCollection([], facecolors='orange', edgecolors='k', alpha=1.0)
    ax.add_collection3d(body)
    ax.add_collection3d(head)

    # Dua halo (garis berputar)
    halo_outer, = ax.plot([], [], [], color='gold', lw=2.5)
    halo_inner, = ax.plot([], [], [], color='yellow', lw=1.8)

    # Kaki
    legs_upper = [ax.plot([], [], [], color='green', lw=3)[0] for _ in range(4)]
    legs_lower = [ax.plot([], [], [], color='darkgreen', lw=3)[0] for _ in range(4)]

    leg_offsets = np.array([
        [15,  20],
        [15, -20],
        [-15,  20],
        [-15, -20],
    ])

    ax.set_xlim(-40, L_TOTAL)
    ax.set_ylim(-WIDTH/2, WIDTH/2)
    ax.set_zlim(-N_STEPS * STEP_HEIGHT - 20, 80)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Makhluk digital buatan Palupi')
    ax.view_init(25, -60)

    frames = 400

    def update(frame):
        arah = 1 if (frame // 200) % 2 == 0 else -1
        t = (frame % 200) / 200
        x_body = t * L_TOTAL if arah == 1 else (1 - t) * L_TOTAL
        y_body = 0
        z_body = z_stair(x_body) + 30

        # Badan utama
        body.set_verts(make_cube(x_body, y_body, z_body, 30))

        # Kepala dengan rotasi
        rot_yaw = 0.5 * np.sin(2 * np.pi * frame / 100)
        head.set_verts(make_cube(x_body, y_body, z_body + 25, 15, rot_yaw))

        # Posisi pusat kepala untuk halo
        xh, yh, zh = x_body, y_body, z_body + 32

        # Dua halo berputar
        theta = np.linspace(0, 2*np.pi, 100)
        rot_angle = np.radians(frame * 2)
        cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)

        # Halo luar
        r_outer = 18
        Xo = xh + r_outer * np.cos(theta) * cos_r - 0 * np.sin(theta)
        Yo = yh + r_outer * np.sin(theta)
        Zo = zh + 5 + np.sin(theta * 3) * 1.2  # sedikit ripple
        halo_outer.set_data(Xo, Yo)
        halo_outer.set_3d_properties(Zo)

        # Halo dalam
        r_inner = 10
        Xi = xh + r_inner * np.cos(theta) * cos_r
        Yi = yh + r_inner * np.sin(theta)
        Zi = zh + 7 + np.cos(theta * 3) * 1.2
        halo_inner.set_data(Xi, Yi)
        halo_inner.set_3d_properties(Zi)

        # Gerakan kaki
        for i in range(4):
            phase = (frame + i * 50) % 200
            step_phase = np.sin(2 * np.pi * phase / 200)

            x_base = x_body + leg_offsets[i,0]
            y_base = y_body + leg_offsets[i,1]

            x_foot = x_base + arah * 15 * step_phase
            z_ground = z_stair(x_foot)
            z_foot = z_ground + 5 + 10 * max(0, step_phase)

            x_knee = (x_body + x_foot) / 2
            z_knee = z_body - 10 + 8 * abs(step_phase)

            legs_upper[i].set_data([x_body, x_knee], [y_body, y_base])
            legs_upper[i].set_3d_properties([z_body - 10, z_knee])

            legs_lower[i].set_data([x_knee, x_foot], [y_base, y_base])
            legs_lower[i].set_3d_properties([z_knee, z_foot])

        return [body, head, halo_outer, halo_inner, *legs_upper, *legs_lower]

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    plt.tight_layout()
    plt.show()

# Jalankan animasi
animasi_robot_spider_halo()
