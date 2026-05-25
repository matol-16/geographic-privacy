from adversarial_eval import evaluate_attack_on_dataset, evaluate_attack_transferability
from pipe_trajectory import PlonkPipelineTrajectory
from plots_adversarial_attacks import plot_results

### Here, you can run the evaluations on the YFCC and OSV-5M datasets, for different attack budgets. 
# You can also plot the results of the evaluations directly, and compare the localizability of the perturbed images for different attack budgets.

#To evaluate an attack:
# - uncomment the corresponding attack_budget lists (not the same for every dataset for computing costs /training time reasons)
# - select train_args, and the corresponding pipeline for the dataset you want to evaluate on (YFCC or OSV-5M)
# - run evaluate_attack_on_dataset, or plot_results directly.


if __name__ == "__main__":
    # download_osv5m_test()
 
    device = "cuda"
 
    attack_budgets = [1/255,2/255,5/255,10/255,20/255,30/255, 50/255] #attack budgets for YFCC evaluation
    # attack_budgets = [2/255,20/255,50/255] #Attack budgets for localizability evaluation
    # attack_budgets = [1/255,2/255,5/255,10/255,15/255,20/255,25/255,30/255, 50/255] #attack budgets for OSV evaluation

    
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
 
    # #plot all attacks together
    plot_results(
        results_dir="./results",
        attack_budgets=attack_budgets,
        plot_dir="./plots",
        dataset_name="yfcc",
        attack_types=["encoder", "diffusion"],
        all_results=None,
        stored_metrics=["final_step_displacement_predicted"])
    
    
    #### Code to evaluate localizability
    
    # attack_budget = 20/255
    # results_attack_budgets = [2/255, 20/255, 50/255]
    # results_dir = "./results"
    # all_datasets_results = {
    #     ds: torch.load(os.path.join(results_dir, f"{ds}_results_localizability.pt"))
    #     for ds in ["yfcc", "osv"]
    # }
    # plot_localizability_results(
    #     attack_budgets=attack_budget,
    #     plot_dir="./plots",
    #     all_datasets_results=all_datasets_results,
    #     results_attack_budgets=results_attack_budgets,
    # )
 