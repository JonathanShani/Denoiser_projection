# Denoiser_projection

The following instructions will guide you through running the graphs in the "Denoiser-based projections for 2-D super-resolution multi-reference alignment" paper.

The dataset for figure 3 is available in: https://github.com/clausmichele/CBSD68-dataset

Before plotting a graph, you should run the code generating the relevant experiments:
* python run_data_for_fig_2_5.py iterations_per_round=5 gpu_num=0 name_exp=visual_output
* python run_data_for_fig_3_4.py iterations_per_round=5 gpu_num=7 name_exp=dataset_all_scales
* python run_data_for_fig_6.py high_res=32 low_res=16 projected=True N=100000 iterations_per_round=5 gpu_num=1 name_exp=noise_influence
* python run_data_for_fig_7.py high_res=32 low_res=16 projected=True sigma=0.125 iterations_per_round=5 gpu_num=1 name_exp=N_influence
* python run_data_for_fig_8.py high_res=128 low_res=64 projected=True gpu_num=0 name_exp=F_influence
* python run_data_for_table.py projected=True iterations_per_round=5 gpu_num=1 name_exp=proj_mom_em_comparison



After that, you can run the "print_graphs.py" file, extracting the results and plotting all the graphs (or some of them, depending on the test you ran).
