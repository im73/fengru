import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest

_transpose_batch_time = decoder._transpose_batch_time

def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
        dtype=inp.dtype, size=array_ops.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)

class TimeSeriesHelper(tf.contrib.seq2seq.Helper):
    def __init__(self, inputs, sequence_length, time_major=False, name=None): 
        # time_major=False means batch_major=True means [batch_size, time_steps, features]
        with ops.name_scope(name, "TimeSeriesHelper", [inputs, sequence_length]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._sequence_length = ops.convert_to_tensor(sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())

            self._zero_inputs = nest.map_structure(
                  lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

            self._batch_size = array_ops.size(sequence_length)

    @property
    def inputs(self):
        return self._inputs
    @property
    def sequence_length(self):
        return self._sequence_length
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])
    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        with ops.name_scope(name, "TimeSeriesHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
        return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        # useless
        with ops.name_scope(name, "TimeSeriesHelperSample", [time, outputs]):
            sample_ids = math_ops.cast(
                math_ops.argmax(outputs, axis=-1), dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):# sample id是无用变量
        with ops.name_scope(name, "TimeSeriesHelperNextInputs",[time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            '''
            # traininghelper
            def read_from_ta(inp):
                return inp.read(next_time)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(read_from_ta, self._input_tas))
            '''
            
            self.out = outputs
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: self.out
            )# 将decoder得到的ouput作为下一个输入 而不是真实值作为下一个输入
            
        return (finished, next_inputs, state)
    