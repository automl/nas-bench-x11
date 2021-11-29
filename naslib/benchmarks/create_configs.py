import argparse
import os
import yaml


def main(args):

    if args.config_type == 'nas':
        folder = f'{args.out_dir}/{args.dataset}/configs/nas'
        os.makedirs(folder, exist_ok=True)
        args.start_seed = int(args.start_seed)
        args.trials = int(args.trials)

        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'search_space': args.search_space,
                'dataset': args.dataset,
                'optimizer': args.optimizer,
                'out_dir': args.out_dir,
                'search': {'predictor_type': args.predictor,
                           'checkpoint_freq': args.checkpoint_freq,
                           'epochs': args.epochs,
                           'budgets': args.budgets,
                           'single_fidelity': args.single_fidelity,
                           'fidelity': args.fidelity,
                           'sample_size': args.sample_size,
                           'population_size': args.population_size,
                           'num_init': args.num_init,
                           'acq_fn_type': 'its',
                           'acq_fn_optimization': args.acq_fn_optimization,
                           'encoding_type': args.encoding_type,
                           'num_ensemble': args.num_ensemble,
                           'num_candidates': args.num_candidates,
                           'k': args.k,
                           'num_arches_to_mutate': args.num_arches_to_mutate,
                           'max_mutations': args.max_mutations,
                           'batch_size': 256,
                           'data_size': 25000,
                           'cutout': False,
                           'cutout_length': 16,
                           'cutout_prob': 1.0,
                           'train_portion': 0.7
                          }
            }
            path = folder + f'/config_{args.optimizer}_{i}.yaml'

            with open(path, 'w') as fh:
                yaml.dump(config, fh)

    else:
        print('invalid config type in create_configs.py')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_seed", type=int, default=0, help="starting seed")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--single_fidelity", type=int, default=20, help="how many epochs to train")
    parser.add_argument("--fidelity", type=int, default=200, help="nas101:108/nas201:200/nas301:97")
    parser.add_argument("--optimizer", type=str, default='rs', help="which optimizer")
    parser.add_argument("--predictor", type=str, default='bananas', help="which predictor")
    parser.add_argument("--acq_fn_optimization", type=str, default='random_sampling', help="random_sampling/mutation")
    parser.add_argument("--num_init", type=int, default=20, help="how many samples to initialize the predictor")
    parser.add_argument("--num_ensemble", type=int, default=1, help="how many meta networks to ensemble")
    parser.add_argument("--num_candidates", type=int, default=40, help="how many arch candidates to propose")
    parser.add_argument("--k", type=int, default=20, help="how many arch candidates to select")
    parser.add_argument("--num_arches_to_mutate", type=int, default=4, help="how many arches are chosen to mutate")
    parser.add_argument("--max_mutations", type=int, default=5, help="how many edit distances to walk")
    parser.add_argument("--encoding_type", type=str, default='adjacency_one_hot', help="adjacency_one_hot/path")
    parser.add_argument("--test_size", type=int, default=30, help="Test set size for predictor")
    parser.add_argument("--train_size_single", type=int, default=5, help="Train size if exp type is single")
    parser.add_argument("--fidelity_single", type=int, default=5, help="Fidelity if exp type is single")
    parser.add_argument("--population_size", type=int, default=20, help="Population size")
    parser.add_argument("--sample_size", type=int, default=10, help="Candidates from the population")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="How often to checkpoint")
    parser.add_argument("--epochs", type=int, default=100000, help="How many search epochs")
    parser.add_argument("--budgets", type=int, default=400000, help="Estimated wall-clock budgets [s]")
    parser.add_argument("--config_type", type=str, default='nas', help="nas or predictor?")
    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench101/nasbench201/darts/nlp")
    parser.add_argument("--experiment_type", type=str, default='single', help="type of experiment")

    args = parser.parse_args()

    main(args)
