"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import mlflow
from providence.dataloaders import BasicDataloaders
from providence.paper_reproductions import NasaDatasets, NasaTransformer, NasaTransformerOptimizer
from providence.training import generic_training
from providence_utils.hyperparameter_sweeper import HyperparameterSweeper

sweeper = HyperparameterSweeper(lr=[1e-2, 3e-3, 1e-3, 3e-4], batch_size=[2**_pow for _pow in range(4, 5, 6)])

experiment_name = "Providence Sweeps Demo"
mlflow.set_experiment(experiment_name)
exp = mlflow.get_experiment_by_name(experiment_name)

train_ds, test_ds = NasaDatasets()

for sweep_params in sweeper.poll_sweeps():
    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_params(sweep_params)

        model = NasaTransformer()
        opt = NasaTransformerOptimizer(model)

        # set the learning rate through the PyTorch backdoor
        for group in opt.opt.param_groups:
            group['lr'] = sweep_params['lr']
        opt = opt._replace(batch_size=sweep_params['batch_size'])
        # changes for these parameter sweeps

        dls = BasicDataloaders(train_ds, test_ds, batch_size=opt.batch_size)

        losses = generic_training(model, opt, dls)

        # log final losses
        mlflow.log_params(
            {
                "final_training_loss": losses.training_losses[-1],
                "final_validation_loss": losses.validation_losses[-1]
            }
        )
