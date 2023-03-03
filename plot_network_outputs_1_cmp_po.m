%% Load Network Data
load("output_1_CMP_PO/lognormal_50_plot_data.mat") % 5 % 10 % 20 % 50 % 100
%load("output_1_CMP_PO/uniform_50_plot_data.mat") % 5 % 10 % 20 % 50 % 100
%load("output_1_CMP_PO/repaired_lognormal_50_plot_data.mat") % 5 % 10 % 20 % 50 % 100

% select dataset
data = lognormal; % lognormal % uniform % grid

%% Visualise Predictions

figure(1); plot(data.t,data.c,"black .","LineWidth",2); grid on; hold on;

v_plot_t = data.t(data.c_NN >= 0);
v_plot_c_NN = data.c_NN(data.c_NN >= 0);
v_plot_t_fail = data.t(data.c_NN < 0);
v_plot_c_NN_fail = data.c_NN(data.c_NN < 0);
figure(2); plot(v_plot_t,v_plot_c_NN,"black .","LineWidth",2); grid on; hold on; 
plot(v_plot_t_fail,v_plot_c_NN_fail,"red .","LineWidth",2); grid on; hold on; 

figure(4); plot(data.c,data.c_NN,"black .","LineWidth",2); 
xlim([min(v_plot_c_NN),max(v_plot_c)]); ylim([min(v_plot_c_NN),max(v_plot_c)]); 
grid on; hold on;
