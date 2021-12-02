import os
import json


def get_results(path, full_valacc=False):
    results_dict = {}
    errors = 0
    for folder in os.listdir(path):
        # Example of folder: shard_1_results
        
        for hash_abbrev in os.listdir(path + '/' + folder):
            # example of hash_abbrev: 4d
            
            for arch_hash in os.listdir(path + '/' + folder + '/' + hash_abbrev):
                # example of arch_hash: shard_1_results/4d/4ded3e14437b466947928bf36baace5f

                result_file = path + '/' + folder + '/' + hash_abbrev + '/' + arch_hash + '/repeat_1/results.json'
                if os.path.exists(result_file):
                    result = json.load(open(result_file))
                else:
                    errors += 1
                    continue

                if full_valacc:
                    valaccs = []
                    for e in range(108):
                        valaccs.append(result['evaluation_results'][e]['validation_accuracy'])
                    results_dict[arch_hash] = valaccs

                else:
                    # this is not being used. It might be useful if we want train accs later
                    results_dict[arch_hash] = {}
                    for metric in ['train_accuracy', 'validation_accuracy', 'test_accuracy']:
                        for e in [4, 12, 36, 108]:
                            key = metric + '_' + str(e)
                            if key not in results_dict[arch_hash]:
                                results_dict[arch_hash][key] = []
                            results_dict[arch_hash][key].append(result['evaluation_results'][e][metric])
    return results_dict, errors

def get_nasbench_results(nasbench, arch_hashes, full_valacc=False):
    results_dict = {}
    for arch_hash in arch_hashes:
        fix, comp = nasbench.get_metrics_from_hash(arch_hash)
        results_dict[arch_hash] = {}
        
        if full_valacc:
            valaccs = []
            for e in [4, 12, 36, 108]:
                valaccs.append(comp[e][0]['final_validation_accuracy'])
            results_dict[arch_hash] = valaccs
            
        else:        
            for metric in ['train_accuracy', 'validation_accuracy', 'test_accuracy']:
                for e in [4, 12, 36, 108]:
                    key = metric + '_' + str(e)
                    if key not in results_dict[arch_hash]:
                        results_dict[arch_hash][key] = []
                    results_dict[arch_hash][key].append(comp[e][0]['final_' + metric])
    return results_dict

def get_nb101_data(data_root):

    # get trained results
    results, errors = get_results(os.path.join(data_root, 'results_may19'), full_valacc=True)
    print('num nb101 arches:', len(results), 'errors:', errors)

    return results