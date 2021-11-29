export PYTHONPATH=$HOME/nas-bench-x11:$PYTHONPATH
export OMP_NUM_THREADS=2
#optimizers=(rs)
optimizers=(re)
#optimizers=(rea_lce)
#optimizers=(rea_svr)
#optimizers=(ls)
#optimizers=(ls_lce)
#optimizers=(ls_svr)
#optimizers=(bananas)
#optimizers=(bananas_svr)
#optimizers=(bananas_lce)
#optimizers=(hb)
#optimizers=(bohb)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

if [[ $optimizers == bananas* ]]
then
  acq_fn_optimization=mutation
else
  acq_fn_optimization=random_sampling
fi

# folders:
base_file=naslib
s3_folder=results/nas201
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=nasbench201
dataset=cifar100
budgets=800000
fidelity=200
single_fidelity=20
population_size=20
sample_size=10
num_init=20
num_arches_to_mutate=4
max_mutations=5

# trials / seeds:
trials=30
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#optimizers[@]}-1)) )
do
  optimizer=${optimizers[$i]}
  python $base_file/benchmarks/create_configs.py \
  --budgets $budgets --start_seed $start_seed --trials $trials \
  --out_dir $out_dir --dataset=$dataset --config_type nas \
  --search_space $search_space --optimizer $optimizer \
  --acq_fn_optimization $acq_fn_optimization \
  --fidelity $fidelity --single_fidelity $single_fidelity --population_size $population_size \
  --sample_size $sample_size --num_arches_to_mutate $num_arches_to_mutate --max_mutations $max_mutations \
  --num_init $num_init
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
  for optimizer in ${optimizers[@]}
    do
      config_file=$out_dir/$dataset/configs/nas/config\_$optimizer\_$t.yaml
      echo ================running $optimizer trial: $t =====================
      python $base_file/benchmarks/nas/runner.py --config-file $config_file
    done
done
