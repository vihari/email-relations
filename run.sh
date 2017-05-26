#!/usr/bin/bash
rm -R ~/sandbox/rmn_epadd
python train_rn.py \
       --number_of_steps=1000 \
       --log_every_n_steps=10
