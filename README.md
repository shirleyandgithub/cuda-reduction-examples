# 归约求和

对长度为 `N` 的 `float` 数组求总和。实现从全局内存折半加到 warp shuffle，再到 `float4` 向量化读取，源码在 [`sum.cu`](./sum.cu)。

## 编译运行

```bash
cd reduce/sum
cmake -B build && cmake --build build
./build/sum
```

`main` 里 `N = 100000000`，`BLOCK_SIZE = 128`，每个版本重复 10 次取平均。计时包住整个 `call_reduce_*`：里面的 `cudaMemcpy`、同步，以及 v0–v2 在主机上的第二次累加。

## 版本一览

| 版本 | block 内怎么加 | 各 block 的结果怎么合成一个数 |
| --- | --- | --- |
| v0 | 直接在全局内存上折半 | 写到 `d_y[blockIdx.x]`，CPU 再加一遍 |
| v1 | 先搬到静态共享内存，再折半 | 同上 |
| v2 | 先搬到动态共享内存，再折半 | 同上 |
| v3 | 同 v2 | `atomicAdd` 加进同一个 `d_y` |
| v4 | warp 内用 shuffle，warp 之间用 32 个 float 的共享内存 | `atomicAdd` |
| v5 | 每个线程先用 `float4` 读 4 个数，其余同 v4 | `atomicAdd` |

折半归约（v0–v3）要求 `BLOCK_SIZE` 是 2 的幂。v4、v5 要求 `BLOCK_SIZE` 是 32 的倍数。

## v0：全局内存

```cpp
__global__ void device_reduce_v0(float* d_x, float* d_y) {
    const int tid = threadIdx.x;
    float *x = &d_x[blockIdx.x * blockDim.x];

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}
```

一个 block 负责 `blockDim.x` 个元素。`offset` 从一半开始，每轮减半：前半段线程把后半段加到自己身上。`tid == 0` 时，`x[0]` 就是这个 block 的部分和。

`GRID_SIZE = CEIL(N, BLOCK_SIZE)`。kernel 结束后把 `d_y` 拷回主机，再把 `GRID_SIZE` 个部分和加起来。

这一版直接改输入数组，没有越界判断，所以 `N` 必须是 `BLOCK_SIZE` 的整数倍。`main` 里同一个 `d_nums` 会被重复归约，第二次读到的已经是被改过的数据，打印出的 sum 会和主机结果差很远。

## v1：静态共享内存

```cpp
template <const int BLOCK_SIZE>
__global__ void device_reduce_v1(float* d_x, float* d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ float s_y[BLOCK_SIZE];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) d_y[bid] = s_y[0];
}
```

静态 `__shared__` 数组的长度必须是编译期常量，所以 block 大小用模板参数 `BLOCK_SIZE` 传进来。

v0 每一轮折半都在全局内存上读写，一共大约 `log2(BLOCK_SIZE)` 轮。v1 只在开头把数据搬进共享内存，后面的折半都发生在共享内存里。全局内存延迟是几百个时钟周期，共享内存是几十个，所以这一步是 v0 到 v1 的主要加速来源。越界线程填 `0.0`，`N` 不再必须整除 `BLOCK_SIZE`，输入数组也不会被改掉。

## v2：动态共享内存

算法和 v1 相同。共享内存改成启动时再决定大小：

```cpp
extern __shared__ float s_y[];

device_reduce_v2<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>(d_x, d_y, N);
```

第三个启动参数是动态共享内存的字节数。kernel 不必再做成模板，换 block 大小不用重新实例化。归约仍然在共享内存里完成，耗时和 v1 处于同一档。

## v3：atomicAdd 去掉 CPU 收尾

block 内仍然是动态共享内存上的折半归约。每个 block 得到 `s_y[0]` 之后，不再写成 `d_y[bid]`，而是：

```cpp
if (tid == 0) {
    atomicAdd(d_y, s_y[0]);
}
```

多个 block 的 0 号线程会同时更新同一个地址。`atomicAdd` 把「读出、相加、写回」做成一次原子操作。如果写成 `*d_y += s_y[0]`，两个 block 可能读到同一个旧值，后写的覆盖先写的，结果会丢。

调用前要把这一个 `float` 清零。主机改不了设备内存，所以先在主机写成 `0`，再 `cudaMemcpy` 到 `d_y`：

```cpp
*h_y = 0.0;
cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);
device_reduce_v3<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>(d_x, d_y, N);
cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
```

## v4：warp shuffle

一个 block 最多 1024 个线程，每 32 个线程一个 warp，所以最多 32 个 warp。共享内存只留 32 个 `float`，用来放每个 warp 的部分和。

```cpp
__shared__ float s_y[32];

int idx = blockDim.x * blockIdx.x + threadIdx.x;
int warpId = threadIdx.x / warpSize;
int laneId = threadIdx.x % warpSize;

float val = (idx < N) ? d_x[idx] : 0.0f;
#pragma unroll
for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
if (laneId == 0) s_y[warpId] = val;
__syncthreads();

if (warpId == 0) {
    int warpNum = blockDim.x / warpSize;
    val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (laneId == 0) atomicAdd(d_y, val);
}
```

同一个 warp 里的线程由硬件锁在一起，shuffle 直接在寄存器之间传值，这一级不写共享内存，也不调用 `__syncthreads()`。`__shfl_down_sync(0xFFFFFFFF, val, offset)` 让当前 lane 拿到「自己往下数 `offset` 个 lane」的 `val`。`offset` 从 16 减到 1，一个 warp 就被加成一个数，留在 lane 0。

lane 0 把这个数写入 `s_y[warpId]`。`__syncthreads()` 要等所有 warp 都写完，后面才能读 `s_y`。

然后只用 block 里的 0 号 warp：前 `warpNum` 个 lane 各拿一个 warp 的部分和，其余 lane 补 0，再 shuffle 一次。lane 0 执行 `atomicAdd`。

下面把 warp 画成 8 个线程，只说明数据被压了几级。实际 `warpSize` 是 32。

```text
原始数据：  每 warp 8 个数 × 若干 warp × 若干 block
│
① warp 内   ── 每个 warp 压成 1 个数
│          [1,2,3,4,5,6,7,8] → 36
▼
中间结果：   每个 warp 1 个数，放进 s_y[warpId]
│          s_y = [36, 100, ...]
│
② block 内  ── 每个 block 压成 1 个数
│          [36, 100] → 136
▼
中间结果：   每个 block 1 个数
│          block0 → 136, block1 → 200
│
③ 跨 block  ── atomicAdd
│          136 + 200 = 336
▼
最终结果：   d_y = 336
```

和 v3 比，差在 block 内部。v3 让全部线程在共享内存里一起折半，每轮都要 `__syncthreads()`。v4 先在每个 warp 的寄存器里折半，整个 block 只同步一次，共享内存从「每个线程一个槽」收成「每个 warp 一个槽」。

以 16 个数、1 个 block、2 个长度为 8 的 warp 为例：

| | v3 | v4 |
| --- | --- | --- |
| 第一级 | 16 个线程在 `s_y[16]` 上折半 4 轮，每轮一次 `__syncthreads()` | warp0：`[1..8] → 36`；warp1：`[9..16] → 100`。这一级没有共享内存读写 |
| 中间结果 | `s_y[0] = 136` | `s_y[0] = 36`，`s_y[1] = 100`，然后同步一次 |
| 第二级 | 已经在 `s_y[0]` | 0 号 warp 再把 `[36, 100]` 加成 `136` |
| 写回 | `atomicAdd(d_y, 136)` | `atomicAdd(d_y, 136)` |

## v5：float4

每个线程连续读 4 个 `float`，全局下标要乘 4，grid 也要按「每线程 4 个元素」缩小：

```cpp
// 每个 block 覆盖 BLOCK_SIZE * 4 个元素
const int GRID_SIZE = CEIL(CEIL(N, BLOCK_SIZE), 4);

int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
float val = 0.0f;
if (idx < N) {
    float4 tmp_x = FLOAT4(d_x[idx]);
    val += tmp_x.x;
    val += tmp_x.y;
    val += tmp_x.z;
    val += tmp_x.w;
}
```

读完之后的 shuffle、`s_y[32]`、`atomicAdd` 与 v4 相同。

```text
原始数据：  每线程 4 个数 × 若干线程 × 若干 warp × 若干 block
│
☆ 向量化读入 ── 每线程 4 个先加成 1 个
│          [1,2,3,4] → 10
│          [5,6,7,8] → 26
▼
① warp 内   ── shuffle，同 v4
│          [10,26,42,58] → 136
▼
② block 内  ── 0 号 warp 再 shuffle，同 v4
▼
③ 跨 block  ── atomicAdd，同 v4
```

用 `N = 16`、`BLOCK_SIZE = 4` 看规模变化（示意；真正跑的时候 `BLOCK_SIZE` 要用 128 这种 32 的倍数）：

| | v4（每线程 1 个） | v5（每线程 4 个） |
| --- | --- | --- |
| grid | `CEIL(16, 4) = 4` 个 block | `CEIL(CEIL(16, 4), 4) = 1` 个 block |
| 线程数 | 16 | 4 |
| 每线程读入 | `d_x[idx]` | `d_x[idx .. idx+3]` |
| 读入后的 `val` | `1, 2, …, 16` | `10, 26, 42, 58` |
| 之后 | warp 内 shuffle → `s_y` → 再 shuffle → `atomicAdd` | 相同 |

访存指令大约变成原来的 1/4，参与归约的线程数也变成原来的 1/4。`FLOAT4` 要求地址 16 字节对齐，`cudaMalloc` 得到的设备指针满足这个要求。`idx < N` 只检查了这 4 个数的起点，`N` 不是 4 的倍数时，最后一组会多读出数组末尾。

## 测试

`N = 100000000`，`BLOCK_SIZE = 128`，输入是 `[-1, 1)` 的随机数。浮点加法顺序不同，各版本的 sum 不会逐位相同。v0 因为原地改输入，重复计时后 sum 已经没有意义。

```text
[reduce_host]: sum = -1392.220947, total_time_h = 383.848877 ms
[reduce_v0]: sum = 12188807.000000, total_time_0 = 31.068247 ms
[reduce_v1]: sum = -1392.776123, total_time_1 = 19.648817 ms
[reduce_v2]: sum = -1392.776123, total_time_2 = 19.483204 ms
[reduce_v3]: sum = -1392.776123, total_time_3 = 15.859097 ms
[reduce_v4]: sum = -1392.792847, total_time_4 = 11.208912 ms
[reduce_v5]: sum = -1392.694214, total_time_5 = 4.105523 ms
```

## 参考

1. 《CUDA 编程基础与实践》，樊哲勇