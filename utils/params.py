import numpy as np


# coefficients for abs condition normalization
final_abs_range = [0.5, 1]
fat_max, fat_min = 9, 1                     # mm
density_max, density_min = 400, 200         # fibres/mm^2
r_max, r_min = 0.0231, 0.007                # m
fb_num_max, fb_num_min = 6.77, 1.098        # log(real num)
angle_max, angle_min = 1.0, 0.0             # normalised
fb_len_max, fb_len_min = 1.15, 0.85         # normalised muscle length
cv_max, cv_min = 4.5, 3                     # m/s
iz_max, iz_min = 0.636, 0.365               # percentage

coeff_f_a = (fat_max - final_abs_range[1] / final_abs_range[0] * fat_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_f_b = final_abs_range[0] / (coeff_f_a + fat_min)
coeff_d_a = (density_max - final_abs_range[1] / final_abs_range[0] * density_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_d_b = final_abs_range[0] / (coeff_d_a + density_min)
coeff_r_a = (r_max - final_abs_range[1] / final_abs_range[0] * r_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_r_b = final_abs_range[0] / (coeff_r_a + r_min)
coeff_fb_a = (fb_num_max - final_abs_range[1] / final_abs_range[0] * fb_num_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_fb_b = final_abs_range[0] / (coeff_fb_a + fb_num_min)
coeff_a_a = (angle_max - final_abs_range[1] / final_abs_range[0] * angle_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_a_b = final_abs_range[0] / (coeff_a_a + angle_min)
coeff_len_a = (fb_len_max - final_abs_range[1] / final_abs_range[0] * fb_len_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_len_b = final_abs_range[0] / (coeff_len_a + fb_len_min)
coeff_cv_a = (cv_max - final_abs_range[1] / final_abs_range[0] * cv_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_cv_b = final_abs_range[0] / (coeff_cv_a + cv_min)
coeff_iz_a = (iz_max - final_abs_range[1] / final_abs_range[0] * iz_min) / (final_abs_range[1] / final_abs_range[0] - 1)
coeff_iz_b = final_abs_range[0] / (coeff_iz_a + iz_min)

coeff_a = {'num': coeff_fb_a, 'depth': coeff_r_a, 'angle': coeff_a_a, 'iz': coeff_iz_a, 'cv': coeff_cv_a, 'len': coeff_len_a}
coeff_b = {'num': coeff_fb_b, 'depth': coeff_r_b, 'angle': coeff_a_b, 'iz': coeff_iz_b, 'cv': coeff_cv_b, 'len': coeff_len_b}

w_amp = 100000

# muscle labels for the nonzeros: ECRB, ECRL, PL, FCU-ulnar head, FCU-humeral head, ECU, ED, FDS
TRAIN_MU_NUM = np.array([0, 140, 153, 123, 154, 163, 135, 0, 0, 0, 0, 0, 140, 117, 0])
ALL_MU_NUM = np.array([0, 186, 204, 164, 205, 217, 180, 0, 0, 0, 0, 0, 186, 158, 0])
TEST_MU_NUM = ALL_MU_NUM - TRAIN_MU_NUM
TEST_MU_NUMS = np.cumsum(TEST_MU_NUM * 256)[[0, 1, 2, 3, 4, 5, 6, 12, 13]]
TEST_MU_NUM = TEST_MU_NUM[[1, 2, 3, 4, 5, 6, 12, 13]]
TRAINSET1 = 145920
TESTSET1 = 48384

# Range of MU depth and MU angle within each muscle
DEPTH = {'ECRB': [0.0130, 0.0220], 'ECRL': [0.0085, 0.0153], 'PL': [0.0071, 0.0114], 'FCU_u': [0.0084, 0.0168], 'FCU_h': [0.0074, 0.0165], 'ECU': [0.0092, 0.0168], 'ED': [0.0079, 0.0171], 'FDS': [0.0169, 0.0231], 'FCU': [0.0074, 0.0168]}
ANGLE = {'ECRB': [0.4946, 0.6632], 'ECRL': [0.4607, 0.5109], 'PL': [0.0540, 0.0956], 'FCU_u': [0.7878, 0.8658], 'FCU_h': [0.9897, 1.0], 'ECU': [0.7194, 0.7779], 'ED': [0.5637, 0.6826], 'FDS': [0.1471, 0.2264], 'FCU': [0.7878, 1.0]}
MS_AREA = {'ECRB': 60.100, 'ECRL': 70.715, 'PL': 35.335, 'FCU_u': 78.925, 'FCU_h': 83.305, 'ECU': 47.720, 'ED': 60.000, 'FDS': 25.335}  # mm^2
