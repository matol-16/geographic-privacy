from adversarial_eval import evaluate_attack_on_dataset, evaluate_attack_transferability
from pipe_trajectory import PlonkPipelineTrajectory
from plots_adversarial_attacks import plot_results

if __name__ == "__main__":
    # download_osv5m_test()
 
    device = "cuda"
 
    attack_budgets = [1/255,2/255,5/255,10/255,20/255,30/255, 50/255]
    # attack_budgets = [2/255,25/255]
    
    attack_budgets = [1/255,2/255,5/255,10/255,15/255,20/255,25/255,30/255, 50/255]

    
    train_args = [{"n_steps":60,
        "train_batch_size":256,
        "lr":1e-3,
        "anchor_samples":512,
        "clean_num_steps":100,
        "target_pure_noise": False,
        "dot_product_loss":"absolute",
        "reconstruction_loss_weight": 0.0,
        "num_restarts" : 5,
        "restart_selection_metric": "final_step_displacement",
        "restart_eval_cfg": 10.0,
        "device": device} for _ in range(len(attack_budgets))]
    
    # pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_OSV_5M_diffusion").to(device)	
    pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_YFCC_diffusion").to(device)

    # evaluate_attack_on_dataset(
    # 	attack_types=["diffusion"],
    # 	dataset_name="yfcc",
    #   	source_image=None, 
    # 	pipeline=pipeline, 
    # 	n_images_to_eval=80,
    # 	attack_budgets=attack_budgets,
    # 	attack_kwargs=train_args,
    #     parallel_workers=7,
    #     use_cuda_streams=True,
    # 	results_dir="./results",
    # 	plot_dir="./plots",
    # 	use_real_gps=False,
    # )
 
    #plot all attacks together
    plot_results(
        results_dir="./results",
        attack_budgets=attack_budgets,
        plot_dir="./plots",
        dataset_name="osv",
        attack_types=["encoder", "diffusion"],
        all_results=None,
        stored_metrics=["final_step_displacement"])
 

    # evaluate_attack_transferability(
    # 	source_image=source_image,
    # 	pipeline=pipeline,
    # 	dataset_name="yfcc",
      # 	n_images_to_eval=100,
    # 	attacks=["diffusion"],
    # 	attack_budgets=attack_budgets,
    # 	attack_kwargs=train_args,
    # 	metric="final_step_displacement",
    # 	results_dir="./results_2",
    # 	plot_dir="./plots_2",
    # )