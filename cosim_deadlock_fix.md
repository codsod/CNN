# CNN_top cosim 死锁问题与修复

## 1. 现象

`step1_simulation.py` 跑 `CNN_top` 时：

- csim 通过：`[SIM 211-1] CSim done with 0 errors.`
- cosim 失败：

```
// ERROR!!! DEADLOCK DETECTED at 15110000 ns! SIMULATION WILL BE STOPPED!
ERROR: [COSIM 212-361] C TB post check failed, nonzero return value '1'.
ERROR: [HLS 200-742] Deadlock detected in co-simulation
ERROR: [COSIM 212-4] *** C/RTL co-simulation finished: FAIL ***
```

日志位置：`instances_tp8_sim/proj_CNN_top/vitis_hls.log:2748-2772`。

## 2. 根因

XSIM 给出的依赖环：

```
Dependence cycle 1:
 (1) do_concat       blocked by empty FIFO concat_i1   (waiting br1)
 (2) do_conv_br_1    blocked by empty FIFO conv_br_i1  (waiting tee3)
 (3) tee3_quant_to_conv_br blocked by FULL  FIFO conv_br_i0
 (4) do_conv_br_0    blocked by FULL  FIFO concat_i0   (concat 没读)
```

死锁形成路径：

1. `case/CNN_top.cpp:62-66` 中 `quant_out / conv_br_i{0,1,2} / concat_i{0,1,2}` 全用默认深度 `hls::stream`（depth=2）。综合 log 第 307–320 行已给出 14 条 `[HLS 200-805] default size can result in deadlock` 警告。
2. `tee3_quant_to_conv_br`（`case/CNN_top.cpp:33-48`）每读一笔输入，需要**同时**写三路 `out0/out1/out2`，任一路写不进去整个 tee3 阻塞。
3. `src/concat.h:104-134` 在每个 `t` 上**严格按 br0 → br1 → br2 顺序**消费输入：先把 br0 的 13 个通道全部 read+写 buf0，再读 br1，再读 br2。读 br0 期间完全不消费 `concat_i1 / concat_i2`。
4. 三路 conv_br 的 kernel 大小差异极大（16 / 32 / 64），输出节奏不同；br1/br2 因为 kernel 大、line-buffer 填得慢，会落后于 br0。在 br0 已经把 `concat_i0`（深度 2）灌满的同时，br1 还没产出第一个像素，concat 又卡在等 br1，4 个进程互相形成环路死锁。

csim 不会触发是因为 csim 用 C 语言模拟流，`hls::stream` 默认是无界 FIFO，所以功能完全正确，纯属 RTL 上 FIFO 深度不足导致的工程缺陷。

## 3. 修复

最小改动：在 `case/CNN_top.cpp` 流声明处加 `#pragma HLS stream depth`：

```cpp
hls::stream<hls::vector<X_T, IN_DIM>> conv_br_i0, conv_br_i1, conv_br_i2;
hls::stream<hls::vector<X_T, IN_DIM>> concat_i0;
hls::stream<hls::vector<X_T, IN_DIM>> concat_i1;
hls::stream<hls::vector<X_T, IN_DIM>> concat_i2;
#pragma HLS stream variable=conv_br_i0 depth=32
#pragma HLS stream variable=conv_br_i1 depth=32
#pragma HLS stream variable=conv_br_i2 depth=32
#pragma HLS stream variable=concat_i0  depth=32
#pragma HLS stream variable=concat_i1  depth=32
#pragma HLS stream variable=concat_i2  depth=64   // br2 输出最慢、通道最多
```

深度按一个 `t` 步内 concat 批量积压的数量级取（br0/br1/br2 通道数为 13/13/14）。

## 4. 备选/更彻底的方案

- **交错读取**：把 `do_concat` 三段 `for(oc<BRk_CHANNELS)` 改成 `(br0 一笔 / br1 一笔 / br2 一笔)` 交错消费，让消费速率匹配生产速率，不再依赖 FIFO 深度。
- **重标定下沉到 concat**：即 `check_ques.md §5` 提到的方向（`src/old/new` 系列实验文件已实现），既能省内存又能改善背压；当时为保证单模块 csim 一致而并存，可作为后续整合方向。
- 若仍出现 backpressure，可同步把 `quant_out` 也加 `depth=32`。

## 5. 验证

重跑 `python3 step1_simulation.py`（注意是 `python3`，本机没有 `python` 别名），观察 `instances_tp8_sim/proj_CNN_top/vitis_hls.log` 是否还出现 `DEADLOCK DETECTED` 与 `COSIM 212-4 FAIL`。
