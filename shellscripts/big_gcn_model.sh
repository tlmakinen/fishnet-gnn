#PBS -N big_gcn
# PBS -q batch
#PBS -j oe
#PBS -o /data101/makinen/jobout/${PBS_JOBNAME}.${PBS_JOBID}.o
#PBS -M l.makinen21@imperial.ac.uk
# PBS -n
#PBS -l nodes=1:has1gpu:ppn=32,walltime=48:00:00

# load env
echo loading env...

module load cuda/11.8
module load intelpython/3-2023.0.0
XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS
source /home/makinen/venvs/fastjax/bin/activate

echo cd-ing...
cd /home/makinen/repositories/fishnet-gnn/

echo running script...
# configs, (fishnet: 0 or gcn: 1), model_size (or num layers), load_model (1; yes, 0: no), (model to load: best or last)
python -m main_train.py ./comparison/configs_main_comparison.json 1 big_model 0 last


echo done...


exit 0