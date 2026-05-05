## 1.pool层的位列问题与取整问题

- 实际算法中，pool层的每一个通道内，T=410 而不是411
- 实际算法中的取整是银行家取整法，主要区别在于0.5，此方法是选择最近的偶数，而四舍五入是直接向上取整

## 2.linear层，最后一步requant没做round

## 3.conv_br层

- 实际算法中，输出通道里T参数并不是411 而是410 ，并且卷积时采用了非对称padding
- 当前源码里，br2 的 output_scale 和 meta 不一致
```
constexpr SCALE_T CONV_BR2_OUTPUT_SCALE = 0.064672186971; // conv2d.h
Output_Scale: 0.064764507115
```

## 4.conv_spatial层

- 没有重标定环节，三个输入量化域被硬性统一了，故而产生偏差
- 银行家舍入
- 以及参数不一致
```
constexpr SCALE_T CONV_SPATIAL_OUTPUT_SCALE = 0.059495586902;
参考 meta conv2_meta.txt  0.059495572001
```

- 真正的问题在这里：

- acc0304.ipynb 里导出 conv2 参数时，conv2 的 input_scale/input_zp 不是从 concat(FloatFunctional.cat) 的真实输出里读出来的，而是直接沿用了最后一路 branch 的 LUT 量化参数。
- 但 trace_conv2.txt 又是从真实量化模型跑出来的输出 trace。
- 于是就出现了“trace_conv2 是真的，conv2_meta.txt/conv2_bias_int32.txt 却是按一个近似假设导出的”这个不一致。

- 目前已经找到正确的requant参数，SCALE为
```
constexpr SCALE_T CONV_SPATIAL_INPUT_SCALE = 0.034158173949;
```
- 同时需要注意的是，在导出bias时，公式中有 requant 对应的SCALE
- 所以这里的bias也需要重新修改

## 5.new文件做了什么事

- 重标定最合适是放在concat中，因为concat存buffer时本来就要展开一次循环，正好完成重标定
- 使内存开销和时间开销都降低
- 但是可能会导致这两个模块的单模块仿真出现问题，因此专门放到new文件中，使两个版本的代码都存在