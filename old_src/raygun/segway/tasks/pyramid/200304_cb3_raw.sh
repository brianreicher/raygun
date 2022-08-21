
f=/n/groups/htem/temcagt/datasets/cb3/zarr/cb3.zarr
ds="volumes/raw"
output_file=$f
roi_offset='0 307200 1095680'
roi_shape='64000 65536 65536'

mkdir -p ${output_file}/${ds}_mipmap
ln -s `realpath ${f}/${ds}` ${output_file}/${ds}_mipmap/s0

# in_ds=${ds}_mipmap/s0
# out_ds=${ds}_mipmap/s1
# scale_factor='1 2 2' # to 40x8x8
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2 --scheduling_block_size_mult 4 4 4

# in_ds=${ds}_mipmap/s1
# out_ds=${ds}_mipmap/s2
# scale_factor='1 2 2' # to 40x16x16
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2 --scheduling_block_size_mult 4 4 4 --config_hash 58222722205986825520707582790391892294

# in_ds=${ds}_mipmap/s2
# out_ds=${ds}_mipmap/s3
# scale_factor='1 2 2' # to 40x32x32
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2 -scheduling_block_size_mult 4 4 4 

in_ds=${ds}_mipmap/s3
out_ds=${ds}_mipmap/s4
scale_factor='1 2 2' # to 40x64x64
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2

in_ds=${ds}_mipmap/s4
out_ds=${ds}_mipmap/s5
scale_factor='2 2 2' # to 80x128x128
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2

in_ds=${ds}_mipmap/s5
out_ds=${ds}_mipmap/s6
scale_factor='2 2 2' # to 160x256x256
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2

in_ds=${ds}_mipmap/s6
out_ds=${ds}_mipmap/s7
scale_factor='2 2 2' # to 320x512x512
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2

in_ds=${ds}_mipmap/s7
out_ds=${ds}_mipmap/s8
scale_factor='2 2 2' # to 640x1024x1024
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2

in_ds=${ds}_mipmap/s8
out_ds=${ds}_mipmap/s9
scale_factor='2 2 2' # to 1280x2048x2048
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 2
