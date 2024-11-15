export CUDA_VISIBLE_DEVICES=0

# Config model_name ------------ #
INPUT=${0##*/}
SUBSTRING=$(echo $INPUT| cut -d'_' -f 2)

INPUT=$SUBSTRING
SUBSTRING=$(echo $INPUT| cut -d'.' -f 1)
# ------------------------------ #


python -u main.py --dataset=ml-1m \
--train_dir=tmp/${SUBSTRING} \
--num_epochs=801 \
--maxlen=200 \
--attention_type=positional \
--dropout_rate=0.2 \
--loss_type=sparse_ce \
