import numpy as np
import pandas as pd
import opensim as osim

import matplotlib.pyplot as plt
import seaborn as sns


# GLOBAL VARIABLES
pose_pth = './MSK/poses.csv'
pose_basis = pd.read_csv(pose_pth)

model_pth = './MSK/models/ARMS_Wrist_Hand_Model_4.3/Hand_Wrist_Model_for_development.osim'
model = osim.Model(model_pth)
all_ms_labels = []
for ms in model.getMuscles():
    all_ms_labels.append(ms.getName())


def pos2mov(poses, durations, fs=50):
    """
    poses: List(str), e.g., ['default', 'default+flex', 'default', 'default+ext', 'default'] denotes a flexion and extension movement
    open, grasp, flex, ext, rdev, udev
    durations: List(double), e.g., [2.0] * 5
    fs: frequency of joint angles in Hz
    """

    assert len(poses) - 1 == len(durations), 'number of poses not match number of durations'
    num_pose = len(poses)

    # Get pd of time and joints
    mov = []
    for i in range(num_pose - 1):
        time_dim = durations[i] * fs

        curP = poses[i].replace('default', 'open').split('+')
        nxtP = poses[i + 1].replace('default', 'open').split('+')

        cur_ang = np.zeros(len(pose_basis))
        for p in curP:
            cur_ang += pose_basis.loc[:, p]
        nxt_ang = np.zeros(len(pose_basis))
        for p in nxtP:
            nxt_ang += pose_basis.loc[:, p]

        mov.append(np.linspace(cur_ang, nxt_ang, num=time_dim))
    mov = np.concatenate(mov)
    time = np.linspace(0, np.sum(durations), np.sum(durations) * fs)
    mov = np.concatenate((time[:, None], mov), axis=1)
    mov = pd.DataFrame(data=mov, columns=['time', *pose_basis.iloc[:, 0].tolist()])

    return mov


def mov2len(mov, ms_labels, normalise=True):

    state = model.initSystem()
    ms_lens = pd.DataFrame(columns=['time', *ms_labels])

    # Get default muscle length for normalisation
    default_pose_label = 'open'
    default_pose = pose_basis.loc[:, default_pose_label]
    ms_len_default = {}
    for dof_id, deg in enumerate(default_pose):
        coordinate = np.radians(deg)
        dof = pose_basis.iloc[dof_id, 0]
        model.updCoordinateSet().get(dof).setValue(state, coordinate)
        model.realizePosition(state)
    model.equilibrateMuscles(state)
    for ms in ms_labels:
        ms_len_default[ms] = model.getMuscles().get(ms).getFiberLength(state)

    # Assign 
    for t_id, t in enumerate(mov.iloc[:, 0]):
        for dof_id, dof in enumerate(mov.columns[1:]):
            coordinate = np.radians(mov.loc[t_id][dof_id + 1])
            model.updCoordinateSet().get(dof).setValue(state, coordinate)
            model.realizePosition(state)
        model.equilibrateMuscles(state)
        cur = {'time': t}
        for ms in ms_labels:
            cur[ms] = model.getMuscles().get(ms).getFiberLength(state)
        ms_lens = ms_lens.append(cur, ignore_index=True)

    if normalise:
        for ms in ms_labels:
            ms_lens.loc[:, ms] = ms_lens.loc[:, ms] / ms_len_default[ms]
    return ms_lens


def pos2params(poses, durations, ms_labels, fs=5):
    mov = pos2mov(poses, durations, fs)
    ms_lens = mov2len(mov, ms_labels, True)

    # Assumption: constant volume
    # If lens change by s (*s times), correspondingly depths will change by 1/sqrt(s) and cvs will change by 1/s.
    # The outputs are normalised scales
    # Use it with a predefined absolute value between 0.5 and 1.0

    depths = ms_lens.copy(deep=True)
    for col in depths.columns[1:]:
        depths.loc[:, col] = 1 / (np.sqrt(depths.loc[:, col]) + 1e-8)
    cvs = ms_lens.copy(deep=True)
    for col in cvs.columns[1:]:
        cvs.loc[:, col] = 1 / (cvs.loc[:, col] + 1e-8)

    return ms_lens, depths, cvs


if __name__ == '__main__':
    # poses = ['default', 'default+flex', 'default', 'default+ext', 'default']
    # durations = [2] * 4
    poses = ['default', 'grasp', 'grasp+flex', 'grasp', 'grasp+ext', 'grasp', 'default']
    durations = [2] * 6
    fs = 5

    # mov = pos2mov(poses, durations, fs)
    # fig = plt.figure()
    # for pos in pose_basis.iloc[:, 0].tolist():
    #     mov.loc[:, pos] = mov.loc[:, pos]
    #     sns.lineplot(data=mov, x='time', y=pos)
    # fig.savefig('./figs/DoFs.jpg')

    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']
    ms_lens, depths, cvs = pos2params(poses, durations, ms_labels)

    # fig = plt.figure(figsize=(10, 6))
    # for ms in ms_labels:
    #     # ms_lens.loc[:, ms] = ms_lens.loc[:, ms] / ms_lens.loc[0, ms]
    #     plt.plot(ms_lens.loc[:, 'time'], ms_lens.loc[:, ms])
    # plt.legend(ms_labels, bbox_to_anchor=(1.0, 0.), loc='lower left')
    # fig.savefig('./figs/muscle_lengths.jpg')
