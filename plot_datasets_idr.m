%% % lognormal
dataset = readtable("output_IDR/dataset_IDR_N_125000_R0_100.0_kout_log-normal[0.5,0.3]_Imax_1.0_IC50_log-normal[2.5,0.3]_WT_uniform[1.0,3.0]_dose_50_Cl_0.2_V_2.0_ka_0.5_tmax_96.0.csv");

%% % uniform
% dataset = readtable("output_IDR/dataset_IDR_N_125000_R0_100.0_kout_uniform[0.14,1.47]_Imax_1.0_IC50_uniform[0.75,5.7]_WT_uniform[1.0,3.0]_dose_50_Cl_0.2_V_2.0_ka_0.5_tmax_96.0.csv");

%% Plots
v_ind_kout = dataset{:,2};
v_ind_IC50 = dataset{:,3};
v_ind_kin = v_ind_kout ./ 100;

figure(1);
tiledlayout(1,3);
nexttile; hist(v_ind_kout); hold on;
nexttile; hist(v_ind_IC50); hold on;
nexttile; hist(v_ind_kin); hold on;
