%% % lognormal
dataset = readtable("output_1_CMP_PO/dataset_1_cmp_po_N_125000_Cl_log-normal[0.2,0.3]_V_log-normal[2.0,0.3]_ka_log-normal[0.5,0.3]_WT_uniform[1.0,3.0]_dose_50_tmax_24.0.csv");

%% % uniform
% dataset = readtable("output_1_CMP_PO/dataset_1_cmp_po_N_125000_Cl_uniform[0.05,0.62]_V_uniform[0.63,6.6]_ka_uniform[0.13,1.38]_WT_uniform[1.0,3.0]_dose_50_tmax_24.0.csv");

%% Plots
v_dose = dataset{:,1};
v_ind_Cl = dataset{:,2};
v_ind_V = dataset{:,3};
v_ind_ka = dataset{:,4};
v_WT = v_dose ./ 50;

figure(101); plot(v_WT,v_ind_V,"black .");

figure(102);
tiledlayout(1,4);
nexttile; hist(v_ind_Cl,100); hold on;
nexttile; hist(v_ind_V,100); hold on;
nexttile; hist(v_ind_ka,100); hold on;
nexttile; hist(v_WT,100); hold on;

fprintf("min(WT) = %8.4f , max(WT) = %8.4f \n",min(v_WT),max(v_WT));
fprintf("min(Cl) = %8.4f , max(Cl) = %8.4f \n",min(v_ind_Cl),max(v_ind_Cl));
fprintf("min(V)  = %8.4f , max(V)  = %8.4f \n",min(v_ind_V),max(v_ind_V));
fprintf("min(ka) = %8.4f , max(ka) = %8.4f \n",min(v_ind_ka),max(v_ind_ka));
