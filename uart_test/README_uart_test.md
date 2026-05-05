# UART Test 工程说明

## 1. 工程用途

这是一个最小化的 FPGA 串口测试工程，主要用于验证以下几件事：

- FPGA 配置是否成功
- 板载时钟是否正常
- USB 转串口链路是否正常
- FPGA 的 UART 发送和接收逻辑是否正常

下载到 FPGA 后，这个工程会先主动发送一串固定字符串，然后等待串口输入，并把收到的数据原样回发。

---

## 2. 上电后的行为

工程顶层是 [uart_test.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_test.v:1)。

上电或重新配置后，逻辑行为如下：

1. 内部先保持一小段复位时间
2. 通过串口发送字符串 `HELLO WORLD\r\n`
3. 进入等待状态
4. 如果 PC 通过串口发送了数据，FPGA 会把收到的字节原样发回
5. 如果一段时间没有收到数据，则再次发送 `HELLO WORLD\r\n`

所以它本质上是：

- 一个“上电打招呼”的串口发送测试
- 一个“收到什么就回什么”的串口回环测试

---

## 3. 顶层接口

当前顶层端口如下，定义见 [uart_test.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_test.v:4)：

- `sys_clk`
  - 板载单端 `50 MHz` 时钟输入
- `uart_rx`
  - FPGA 串口接收脚
  - 接收来自 USB 转串口芯片的数据
- `uart_tx`
  - FPGA 串口发送脚
  - 发送数据到 USB 转串口芯片

当前版本没有使用外部复位输入，内部使用一个简单的上电复位计数器生成 `rst_n`。

---

## 4. 模块组成

工程一共 3 个主要 Verilog 模块：

### `uart_test.v`

顶层控制模块，负责：

- 产生上电内部复位
- 控制发送 `HELLO WORLD\r\n`
- 调用 UART 接收模块
- 调用 UART 发送模块
- 在收到串口数据后执行回发

### `uart_tx.v`

串口发送模块，负责把 8bit 并行数据按照 UART 格式发出去。

参数见 [uart_tx.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_tx.v:3)：

- `CLK_FRE`
  - 时钟频率，单位 MHz
- `BAUD_RATE`
  - 串口波特率

当前顶层实例化时设置为：

- `CLK_FRE = 50`
- `BAUD_RATE = 460800`

### `uart_rx.v`

串口接收模块，负责从串口线上采样并恢复出 8bit 数据。

参数见 [uart_rx.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_rx.v:3)，同样为：

- `CLK_FRE = 50`
- `BAUD_RATE = 460800`

---

## 5. 顶层状态机说明

顶层状态机在 [uart_test.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_test.v:43)。

状态定义：

- `IDLE`
  - 空闲状态
  - 上电后会立刻进入 `SEND`

- `SEND`
  - 逐字节发送字符串 `HELLO WORLD\r\n`
  - 发送完成后进入 `WAIT`

- `WAIT`
  - 等待串口输入
  - 如果收到了 1 个字节，就把这个字节送给发送模块回发
  - 如果等待约 1 秒还没有收到数据，则重新进入 `SEND`

字符串内容由组合逻辑生成，见 [uart_test.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_test.v:90)。

发送内容依次为：

- `H`
- `E`
- `L`
- `L`
- `O`
- 空格
- `W`
- `O`
- `R`
- `L`
- `D`
- `\r`
- `\n`

说明：

- 代码里的注释写的是“Send 12 bytes data”，但实际 case 列表包含 13 个字符
- 状态机仍然能正常工作，只是这类注释不够严谨

---

## 6. UART 参数

当前工程采用标准异步串口格式：

- 波特率：`460800`
- 数据位：`8`
- 校验位：`None`
- 停止位：`1`
- 流控：`None`

如果用串口调试助手观察现象，需要按这个参数配置。

---

## 7. 板卡引脚映射

约束文件见 [uart_test.xdc](/Users/linfeng/Downloads/uart_test/uart_test.srcs/constrs_1/new/uart_test.xdc:1)。

当前按 `FACE_K7_SVIC_V10_250916.pdf` 做了如下映射：

- `sys_clk` -> `U27`
  - 网络名：`FPGA_SYS_CLK_50M`
  - 单端 `50 MHz`

- `uart_rx` -> `W22`
  - 网络名：`FDT0_FPGA_TXD`
  - 含义：USB 转串口芯片发送，FPGA 接收

- `uart_tx` -> `W21`
  - 网络名：`FDT0_FPGA_RXD`
  - 含义：FPGA 发送，USB 转串口芯片接收

I/O 电平标准：

- `sys_clk`：`LVCMOS33`
- `uart_rx`：`LVCMOS33`
- `uart_tx`：`LVCMOS33`

---

## 8. 目标器件

工程当前目标器件见 [uart_test.xpr](/Users/linfeng/Downloads/uart_test/uart_test.xpr:11)：

- `xc7k325tffg900-2`

对应你的板上 FPGA：

- `XC7K325T-2FFG900I`

说明：

- 工程里用的是 Vivado 常见 part 写法
- 你的实物是工业级 `I` 版本
- 如果本机 Vivado 器件库里要求更精确型号，可以在 Vivado 中手动切到对应工业级 part

---

## 9. 如何验证功能

最简单的方法是使用串口调试助手。

### 步骤

1. 给 FPGA 下载 bit 文件
2. 用 USB 连接板卡与电脑
3. 打开串口调试助手
4. 选择对应串口号
5. 配置串口参数：
   - `460800`
   - `8N1`
   - 无流控
6. 打开串口

### 预期现象

- FPGA 下载完成后，串口窗口应出现：

```text
HELLO WORLD
```

- 手动发送一个字符，比如 `A`
- FPGA 应该回发 `A`

如果持续不输入数据，工程会周期性再次发送 `HELLO WORLD`

---

## 10. 代码阅读重点

如果其他 AI 或工程师要快速理解这个工程，建议优先看这几个位置：

- 顶层行为入口：
  - [uart_test.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_test.v:43)

- `HELLO WORLD` 字符生成：
  - [uart_test.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_test.v:90)

- UART 发送状态机：
  - [uart_tx.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_tx.v:28)

- UART 接收状态机：
  - [uart_rx.v](/Users/linfeng/Downloads/uart_test/uart_test.srcs/sources_1/src/uart_rx.v:45)

- 管脚约束：
  - [uart_test.xdc](/Users/linfeng/Downloads/uart_test/uart_test.srcs/constrs_1/new/uart_test.xdc:1)

---

## 11. 一句话总结

这是一个运行在 `XC7K325T-2FFG900I` 板卡上的最小 UART 测试工程：

- 板载 `50MHz` 时钟驱动
- 上电发送 `HELLO WORLD\r\n`
- 串口收到什么就回发什么
- 适合用于验证 FPGA 下载、时钟、UART 引脚和 USB 转串口链路是否正常
