#!/bin/bash

for i in {0..9..1}
do
  f="/home/andre/evolved_data/resnet18_.layer4.1.Conv2dconv2_unit"
  f+=$i
  f+="/inh/"
  new_f=$f
  new_f+="no_mask/"
  mkdir $new_f

  for j in {0..8..1}
  do
    img="Best_Img_resnet18_.layer4.1.Conv2dconv2_"
    img+=$j
    img+=".png"
    old_f=$f
    old_f+=$img
    new_img=$new_f+$img
    mv $old_f $new_f
  done
done

#for i in {0..9..1}
#do
#  old_f="/home/andre/rank_data/rand_abl/alexnet_trained/unit"
#  old_f+=$i
#  old_f+="_rand_abl_top9_ranks.npy"
#
#  new_f="/home/andre/rank_data/rand_abl/alexnet_trained/layer11_unit"
#  new_f+=$i
#  new_f+="_top9_ranks.npy"
#
#  mv $old_f $new_f
#
#  old_f="/home/andre/rank_data/rand_abl/alexnet_trained/unit"
#  old_f+=$i
#  old_f+="_rand_abl_bot9_ranks.npy"
#
#  new_f="/home/andre/rank_data/rand_abl/alexnet_trained/layer11_unit"
#  new_f+=$i
#  new_f+="_bot9_ranks.npy"
#
#  mv $old_f $new_f
#
#  old_f="/home/andre/rank_data/rand_abl/alexnet_trained/unit"
#  old_f+=$i
#  old_f+="_rand_abl_proto_ranks.npy"
#
#  new_f="/home/andre/rank_data/rand_abl/alexnet_trained/layer11_unit"
#  new_f+=$i
#  new_f+="_proto_ranks.npy"
#
#  mv $old_f $new_f
#
#  old_f="/home/andre/rank_data/rand_abl/alexnet_trained/unit"
#  old_f+=$i
#  old_f+="_rand_abl_antiproto_ranks.npy"
#
#  new_f="/home/andre/rank_data/rand_abl/alexnet_trained/layer11_unit"
#  new_f+=$i
#  new_f+="_antiproto_ranks.npy"
#
#  mv $old_f $new_f
#done