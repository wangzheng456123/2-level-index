### Parafilter-CUDA:

- This repo contain beta version of CUDA version parafilter.

#### BUILD:

- To build the project, simply run the build.sh script.


##### Requirements:

- C++ 17 or higher

- CUDATookit 12.0+

- CMake >= 3.26.4

#### Run parafilter:

1. ##### parse dataset:

   Parafilter uses an HDF5 dataset as input. Before running Parafilter, you need to parse the HDF5 dataset into raw data. This can be done by simply running the read_hdf5.py script located in the src folder.

2. ##### config parafilter: 
   Before running Parafilter, you need to create a parafilter.conf file. A sample configuration file is shown below:

```c
path = "dataset/amazon/"
pq_dim = 4
n_clusters = 256
exp1 = 120
exp2 = 120
topk = 100
break_down = 1
mem_bound = 5294967296
enable_multi_gpu = 1
is_calc_mem_predictor_coeff = 0
merge_rate = 0.035
lowest_query_batch = 125
```

Here are some notes for the configuration:

- **break_down**: Specifies whether to log performance data.

- **mem_bound**: Determines the appropriate query and data batch sizes. It indicates the upper bound of the GPU memory used by Parafilter.

- **is_calc_mem_predictor_coeff**: Parafilter requires certain configuration parameters to determine the optimal query and data batch sizes. If set to `1`, Parafilter will calculate these parameters without performing the actual ANN search. You should set this to `1` and run Parafilter whenever you modify the configuration file before starting the actual ANN search.

- **path**: Specifies the relative path to the dataset. The path is relative to the build folder where the Parafilter binary is located.

The meanings of other configuration parameters correspond directly to the Parafilter algorithm.

3. ##### config filter:
   The filter.conf file is used to configure filters for the ANN search. A sample file is shown below:
```c
l = 5
filter0:
	type = 0
	shift_val = [604800, 0]
filter1:
	type = 0
	shift_val = [30, 30]
filter2:
	type = 0
	shift_val = [107321744, 107321744]
filter3:
	type = 1
	interval_map = [1, 2, 1, 2, 3, 4, 3, 5, 4, 5]
filter4:
	type = 2
	div_value = [10]
```
Parafilter supports three types of filters:

- **Shift type**: `type = 0` for this filter, `shift_val = [left_shift, right_shift]`. The data is filtered by `label - left_shift <= data <= label + right_shift`.

- **Map type**: `type = 1`, each data is filtered by `interval_map[2 * label] <= data <= interval_map[2 * label + 1]`.

- **Div type**: `type = 2`, data is filtered by `data >= label / div_value`.

4. ##### calculate coeficients :
Set is_calc_mem_predictor_coeff to 1 and run parafilter. After that, set it back to 0 to begin the actual ANN search.