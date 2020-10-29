vai_q_tensorflow quantize \
  --input_frozen_graph frozen_bisenet_v1_0.pb \
  --input_nodes input \
  --input_shapes ?,448,704,3 \
  --output_nodes resnet_v1_0/predictions/Reshape_1 \
  --input_fn bisenet_v1_0_input_fn.calib_input \
  --method 1 \
  --gpu 0 \
  --output_dir ./quantize_results \

