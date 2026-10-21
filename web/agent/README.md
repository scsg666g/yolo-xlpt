# 算力监控与可选远程训练 Agent

中心平台的「算力资源 → 添加服务器」填写名称、地址和备注，保存后下载 `monitor_agent.py` 与专属 `agent-config.json`。地址字段用于展示，不是 SSH 连接地址；中心地址需能从训练服务器访问。

Agent 默认仅采集 GPU、计算进程和主机信息。只有在远程配置显式启用训练并设置允许目录后，才会从中台领取训练任务、运行配置好的 YOLO CLI 并回报指标与日志。Agent 使用 Python 3.9+ 标准库，无第三方 Python 依赖，但当前实现依赖 Linux/POSIX 的 `fcntl`、进程组信号和 `/proc`，不应当作 Windows 通用脚本。训练所需依赖由远程 `yolo` 环境提供。GPU 显存单位为 MB，主机内存为字节。

页面操作见 [Web 说明](../README.md)，中心安装见 [项目首页](../../README.md)，双实例和备份见 [部署说明](../deploy/README.md)，已验证的问题见 [项目分析](../../docs/project-analysis.md)。

## 验证与启动

把两个下载文件放在远程服务器的同一目录，然后执行：

```bash
chmod 600 agent-config.json
python3 monitor_agent.py --config agent-config.json --check-config
python3 monitor_agent.py --config agent-config.json --once
python3 monitor_agent.py --config agent-config.json
```

`--check-config` 只在本机检查配置、YOLO 程序、输出写入、状态目录和共享锁，并显示解析后的工作目录、访问目录和输出目录，不连接中心、不打印 Token。`--once` 执行一次算力采样与上报，不领取训练任务；成功返回 0，失败返回非零退出码。持续运行时连接失败会退避重试；网络恢复后上报新采样。Agent 不会补传过期算力数据。终端关闭会结束前台 Agent，长期使用建议配置 systemd。

配置示例（真实 Token 只在创建或重置时提供一次）：

```json
{
  "_comments": {
    "center_url": "填写远程服务器可访问的中心地址；使用代理时填写代理入口",
    "training_cwd": "相对于本配置文件所在目录；省略时使用配置文件目录",
    "training_output_roots": "允许写入训练结果的目录；相对路径基于 training_cwd"
  },
  "center_url": "http://<中心服务器地址>:18088",
  "node_id": "平台生成的节点ID",
  "token": "平台生成的独立Token",
  "report_interval_seconds": 5,
  "nvidia_smi_bin": "nvidia-smi",
  "enable_training": false,
  "allowed_training_roots": [],
  "training_output_roots": [],
  "yolo_bin": "yolo"
}
```

## 启用远程训练

当前脚本为 **Agent 2.0.1**，配套开发平台为 **1.3.0**，并要求 `center_url` 能从远程机器访问。中心仍将 1.8.0 作为新远程训练的最低版本；进程关联另需 Agent 1.9.0。官方 Agent 从 2.0.0 开始上报 GPU 并行能力，但中心实际以 `training.gpu_parallelism=true` 为调度开关，版本号本身不会代替能力上报。以下表格保留协议能力的版本演进，不表示更旧 Agent 仍具备当前版本的全部调度能力：

| 能力 | 最低 Agent 版本 | 补充条件 |
| --- | --- | --- |
| 基础远程训练、状态、指标与日志尾部 | 1.2.0 | 显式启用训练，设置环境和允许目录 |
| 远程断点续训、产物扫描传输及永久删除 | 1.3.0 | 检查点或目标文件可用，节点保持接入 |
| 多 Agent 共享训练锁 | 1.4.0 | 必须使用同一锁文件，不能仅凭版本保证互斥 |
| 远程 YOLO 训练过程解析 | 1.5.0 | 仅目标检测任务；按批次采样真实输入、层统计和损失摘要 |
| 训练领取租约与启动确认 | 1.6.0 | 未确认领取在 90 秒后可重新分发，降低响应丢失导致的长期“启动中” |
| 确认重试、活动运行恢复与单实例 | 1.7.0 | 配合新版中心处理确认响应丢失、确认前停止和 Agent 异常退出 |
| 训练就绪门禁、层级输出根锁、共享根删除保护和恢复结果判定 | 1.8.0 | 中心按版本、`last_seen_at` 和 `training.ready` 控制训练入口；同机多 Agent 必须使用同一专用系统账户才能共享完整 workspace 保护 |
| 相对路径配置、中文字段说明和输出目录能力上报 | 1.8.1 | 开发平台使用上报目录帮助选择并校验远程输出路径；旧中心可继续接入 Agent，但不会自动获得新版页面和校验逻辑 |
| GPU 进程与训练 Run 关联 | 1.9.0 | Agent 给受管训练进程传递非敏感运行标识，并通过 Linux `/proc` 检查 GPU PID 的父进程链；中心再核验平台实例、节点、Run 和任务。环境不可读、旧进程或链路不完整时保留“无法确认”，不按 PID 猜测 |
| 同节点按 GPU 并行与多 Run 恢复 | 2.0.0 | 上报 `training.gpu_parallelism=true`，且新中心下发有效 `gpu_assignment`；按稳定 GPU UUID 加宿主机锁，同卡互斥，多卡整组获取；单个 Agent 可管理并恢复多个活动 Run |
| 旧版平铺 Attempt 精细清理 | 2.0.1 | 单独删除历史 Attempt 001 时只清理任务根级旧产物，保留所有 `attempt-NNN` 子目录；新版标准 Attempt 的删除协议保持兼容 |

当前中心在创建、启动、续训、重新训练和领取阶段都会拒绝低于 1.8.0 的 Agent。只需监控的旧 Agent 仍可上报算力，但不应用于新的远程训练。下例假设 Agent 配置目录与项目目录 `yolo` 是兄弟目录，按服务器实际布局调整：

```json
{
  "enable_training": true,
  "training_cwd": "../yolo",
  "allowed_training_roots": ["models", "datasets", "runs"],
  "training_output_roots": ["runs"],
  "yolo_bin": "yolo",
  "enable_training_trace": true
}
```

这些字段须加到现有 JSON 对象中，保留原来的 `center_url`、`node_id`、`token` 等字段。配置和示例文件用 `_comments` 对象提供中文说明，它是普通 JSON 元数据，不参与训练；不要改写成带 `//` 或 `/* */` 的 JSONC。

Agent 1.8.1 及以上按下表解析路径，不依赖启动终端所在目录；原有绝对路径配置仍兼容：

| 字段 | 相对路径的基准 |
| --- | --- |
| `training_cwd` | `agent-config.json` 所在目录；省略时就是该目录 |
| `allowed_training_roots`、`training_output_roots`、`training_lock_file` | 解析后的 `training_cwd` |
| `yolo_bin` | 单独写 `yolo` 时通过运行环境的 `PATH` 查找；带 `/` 的相对路径基于 `training_cwd` |

`allowed_training_roots` 应仅列模型文件、数据 YAML、输出根目录所在的最小目录；输出根目录须事先存在，运行 Agent 的用户须有相应写权限。训练产物位于 `<输出根目录>/<任务类型>/<任务 ID>/attempt-001`。启动前 Agent 检查模型文件和数据 YAML 是否存在、路径是否落在允许目录内；它不会逐一检查 YAML 引用的全部图片路径是否也在允许目录内。命令通过参数数组执行，不执行任意 Shell。建议用具有训练目录写入权的专用非 root 用户运行 Agent。

`training_output_roots` 明确哪些允许目录可作为训练输出，每一项必须位于 `allowed_training_roots` 内。旧配置未填写时 Agent 会兼容运行，但只能判断允许目录中是否至少有一个可写；每次启动仍会对任务实际输出根目录再做写入检查。

例如 `runs` 与 `ultralytics/runs` 是项目下的两个独立目录，将后者加入访问目录并不会自动把它加入输出目录。Agent 1.8.1 在 `training.output_roots` 中回传解析后的输出范围，新版开发平台据此提示可用目录并在接受训练前校验路径；最终文件存在性、权限、符号链接和目录边界仍由远程 Agent 检查。1.8.0 未上报该字段时保持兼容。原有任务与 Attempt 的路径快照不会随 Agent 配置自动迁移，应逐一核对；为历史产物保留访问目录，也不代表允许新训练写入该目录或允许绕过共享 owner 删除保护。

`training_lock_file` 仍可选，默认位于 `training_cwd/.yolo-forge-training.lock`。Agent 1.8.x / 1.9.x 和没有收到 `gpu_assignment` 的兼容任务会独占该 workspace 锁以及层级输出根锁，相同或父子输出根因此互斥。中心下发有效 GPU 分配时，Agent 2.0.0 对 workspace 锁和整套层级输出根锁使用共享模式，由 GPU UUID 独占锁控制同卡互斥；不同 GPU 的 Run 可以共享配置输出根，并写入各自独立的任务和 Attempt 目录。删除操作继续独占重叠输出范围。所有训练锁都由 YOLO 子进程继承，Agent 意外结束时不会在训练存活期间提前释放。同一系统用户下的多个 Agent 通过用户私有状态目录共享这些协调锁，不需要为了新版 GPU 任务手工填成相同的 `training_lock_file`；混合新旧中心或旧协议任务仍须保留能够覆盖它们的传统互斥配置。

workspace 登记以 `node_id` 的稳定哈希作为 owner，不依赖配置文件路径或进程 PID。永久删除前，Agent 会在持有层级锁的情况下检查共享根目录的其他存活 owner；只要另一个 Agent 仍声明与删除目标重叠的输出根，就拒绝删除并返回可重试错误。Agent 2.0.0 还以 `nvidia-smi` 返回的稳定 GPU UUID 建立宿主机 `flock`：单卡任务独占对应 UUID，多卡任务按 UUID 排序后一次获取整组锁，避免同卡并发和交叉等待；新中心下发 GPU 分配后，如果设备 UUID 缺失或与当前编号映射不匹配，Agent 会拒绝该 Run，不会退回仅凭设备编号加锁。所有协调状态都位于权限为 `0700` 的用户私有状态目录，因此同机多 Agent 必须使用同一专用系统账户才能共享 GPU 锁、层级锁和删除 owner 保护。不同账户无法获得这套主机级协调。中心本机调度器和平台外启动的 YOLO 进程不受远程 Agent 锁约束。两套平台仍需各自的节点 ID、Token 和 Agent 配置。

在新建任务弹窗选择已注册的远程服务器，再输入远程机器上真实的模型文件、数据集 YAML 和输出根目录的绝对路径。数据集须已在远程服务器可用；中台不会上传或复制它们。中心以自己记录的 `last_seen_at` 判定在线，不使用远程采样时钟作为连通性依据；同时要求 Agent 版本至少为 1.8.0，且最近上报中 `training.enabled=true` 与 `training.ready=true`。任一条件不满足时，创建、启动、续训、重新训练和领取都会返回具体原因；Agent 自身也会暂停领取。Agent 2.0.0 会独立上报 `training.gpu_parallelism=true`，表示协议支持按 GPU 并行；中心还会检查训练就绪状态、设备清单中的 GPU UUID 和资源预约后才下发有效 `gpu_assignment`。远程 GPU 缺少 UUID 时任务继续排队，Agent 启动前若发现编号映射变化则拒绝启动。官方旧 Agent 或缺少能力字段的节点一次只领取一项训练。CPU 和 MPS 任务不占中心 CUDA GPU 预约，但远程节点仍通过节点级独占预约一次处理一项受管训练。指标和最近 100 行日志回传到中台，完整日志与权重保存在远程机器。

Agent 1.5.0+ 会在远程目标检测训练中自动采集最多 24 个真实 Batch 的输入张量、网络层统计、损失和采样图片，并增量上传到中台的过程解析接口。平台不会上传完整特征图、模型权重或数据集；采样图片单张最多 512 KiB。采集失败不会中断训练，网络暂时不可用时会在下一次轮询重试。若远程服务器资源紧张，可在配置中设置 `"enable_training_trace": false`，关闭过程采集但保留普通训练、指标和日志回传。替换 Agent 后需重新领取或重新启动训练，已经运行的旧 Attempt 不会补采历史 Batch。

中心从源码配置动态生成参数目录，Agent 2.0.1 与平台 1.3.0 的发布目录一致。Agent 参数序列化只接受标量，列表和字典会被拒绝；后续中心目录变化时仍需同步 Agent。配置训练后应核对实际命令与产物中的 `args.yaml`，并确认远程 YOLO 版本支持所选参数。

### 远程任务与文件操作（Agent 1.3.0）

远程任务使用与本机任务相同的重命名、克隆、重新训练、回收站和历史运行入口。克隆保留执行服务器和远程路径，不复制数据集或权重。

- **断点续训**：任务须已停止或中断，Agent 已同步该运行的 `weights/last.pt`。续训生成新 Attempt，保留旧运行的文件和指标；使用检查点内的优化器与训练状态。总轮数和数据集必须保持与上次运行一致；想改总轮数或数据集时使用重新训练。
- **产物列表、图片和下载**：Agent 周期上报文件元数据，打开历史运行会排队扫描。预览或下载时才从对应服务器分块传输到中台缓存，校验大小和 SHA256 后提供文件。断网时显示准备中或重试，不会读取中台上的同名远程路径。中心接受的单文件元数据大小上限为 8 GiB；扫描发现超过 1024 个文件会报错，不是只同步前 1024 个。缓存位于中台 `PLATFORM_DATA_ROOT/remote-artifacts/`，永久删除任务时一并清理。
- **移入回收站和恢复**：仅隐藏或恢复平台记录，不立即删除远程文件。活动任务须先停止。
- **永久删除**：提交后显示“远程清理中”，由 Agent 清理该任务各次 Attempt 和对应独立日志，成功确认后中台才删除任务记录。离线时保留记录等待连接；失败显示原因，允许重试清理。提交永久删除后不能恢复，包括已经部分清理的失败任务。Attempt 记录的输出根目录不存在、允许目录失效或磁盘挂载离线时，Agent 会返回可重试失败，不会把“无法核验”当成“已删除”。
- **单独删除 Attempt**：对比页可删除已停止或已结束的一次运行，同时保留任务、其他 Attempt 和已持久化的模型检测历史。新版 `attempt-NNN` 目录按目录清理；Agent 2.0.1 对旧版平铺 Attempt 001 只清理任务根级旧产物，并保留所有 `attempt-NNN` 子目录。远程清理失败时中心保留记录并允许重试。

文件操作仅限配置允许目录内、精确匹配任务及 Attempt 的路径，拒绝路径逃逸及链接目录；清理前核对全部目标，保护训练项目、YOLO 程序以及平台已登记的模型和数据 YAML 输入。没有向平台登记的外部文件引用不在这份保护清单内，应避免把共享训练输入放在任务输出目录中。文件操作通过独立工作线程执行，不阻塞算力心跳或停止指令查询。

### 模型测试历史与 Agent 边界

使用远程权重新建模型图片测试时，中心通过现有产物传输协议准备权重；实际推理、原图和结果持久化都在中心执行。检测完成后，即使远程 Agent 离线，中心保存的历史仍可查看和下载。模型测试历史没有为本轮单独新增 Agent 协议，但当前中心的新远程训练入口统一要求 Agent 1.8.0。

任务移入回收站不会要求 Agent 删除文件，中心模型测试历史也会保留。永久删除远程任务时，Agent 仍须在线清理远程 Attempt 和独立日志；中心保存的模型测试历史属于整套永久删除流程，清理全部成功后才最终删除任务记录。清理失败会保留可重试状态。

### 日志、断网与停止原因

完整日志独立保存到 `<输出根目录>/.platform-logs/YYYYMMDD/<任务 ID>/attempt-NNN.log`，日期由中台下发。例如：

```text
<输出根目录>/.platform-logs/20261012/<任务ID>/attempt-004.log
```

权重、`results.csv`、训练图片仍在原来的 Attempt 产物目录。独立日志不会被 YOLO 双 GPU 的 DDP 启动目录清理删除。新 Attempt 禁止覆盖已有产物或日志；旧 Attempt 路径保持原样，已删除的旧日志无法恢复。

训练进程启动后，即使首次状态上报超时或后续网络中断，Agent 仍保持训练运行。网络恢复后重传最近日志和未确认指标；超过 256 个 Epoch 的积压指标分批补传，中台按运行 ID 和 Epoch 去重。日志尾部是最近 100 行，完整输出需在远程日志文件中查看。

页面日志窗口的“复制全部”和“下载日志”都只处理已回传的尾部，不能取回完整远程日志。独立 `.platform-logs` 目录在 Attempt 产物目录之外，普通产物扫描也不会把这些日志作为训练产物上传。完整输出应在执行服务器按上述路径获取；两个按钮的已知问题见 [日志查看与导出](../README.md#日志查看与导出)。

Agent 1.6.0 开始在协议中使用随机租约，并在启动 YOLO 前向中心确认。若领取响应没有到达 Agent，中心会在 90 秒租约过期后重新分发；过期租约不能继续更新运行，避免旧响应导致重复接管。1.2.0～1.5.x 没有训练领取租约，这只是历史协议说明；当前中心的 1.8.0 最低版本门禁会拒绝它们领取新训练。

Agent 1.8.0 保留 1.7.0 的同租约确认重试与确认前停止保护。网络响应丢失时不启动 YOLO，而是保留活动记录后继续确认；平台明确拒绝过期租约时才放弃。用户在领取和确认之间点击停止时，中心会确认租约并返回 `stop=true`，Agent 不创建训练进程，直接回传已停止。

训练结束时先将最终状态、日志尾部及未确认指标写到 **Agent 配置文件旁的 `.training-state/<运行 ID>.json`**，收到中台确认后才删除。Agent 重启后会按运行分别重试这些文件。此目录不含 Token，但含训练日志，应保持私有权限，并在升级时保留；认证或路径错误需修正后才能补传。

平台停止指令和 Agent 退出会分别显示对应原因，不再统一标记为用户停止。平台停止事件记录请求时间、运行 ID 和连接来源 IP；经网关访问时可能记录网关地址，来源 IP 不能直接代表具体操作者。网络失联不自动判定训练失败或停止。

Agent 将每个活动任务的启动命令、PID、Linux 进程启动时刻、命令行哈希、boot ID 和 GPU 分配写入 `.training-state/active/`，不写入 Token。强制结束 Agent 后多个训练进程仍存活时，Agent 2.0.0 会逐项校验身份并恢复所有匹配的 Run；身份不完整或冲突的记录不会被接管，也不会触发重复训练。接管的进程自然结束后，Agent 使用已完成 Epoch、`best.pt` / `last.pt` 和日志完成标记综合判定：证据足够时回传 `completed`，无法确认时回传 `interrupted` 并保留产物供核对。正常收到 SIGINT/SIGTERM 时 Agent 会向所有受管训练发送停止信号，等待各线程收尾并分别回传状态。

远程任务停止后长时间保持 `stopping` 时，先登录对应训练服务器，根据 Attempt、命令和 PID 确认该 YOLO 进程已经结束。停止状态持续至少 30 秒后，当前训练画布才显示“解除占用”；再按页面提示明确确认远程进程已停止。该操作只将当前 Attempt 标记为 `interrupted` 并解除中心的节点占用，**不会向远程进程发送终止信号**。若旧进程仍在运行就解除占用，新任务可能与它同时占用 GPU。当前 Attempt 已变化或停止状态刚刚更新时，中心会拒绝这次操作，应刷新后重新核对。

每台远程服务器可以配置不同的 `training_cwd`、`allowed_training_roots` 和 `training_output_roots`，平台不会把某一台服务器的绝对路径当作所有节点的固定路径。`training_cwd` 未设置时取配置文件所在目录，不依赖某台服务器的预设项目路径。训练、产物扫描和下载始终按当前 Agent 的允许目录严格校验。永久删除时，不存在的 Attempt 只有在其输出根目录仍位于当前可用的允许目录内时才视为已清理；配置外路径、允许目录消失或挂载离线都会返回可重试错误，不删除中心记录。

### 更新 Agent

先确认没有训练运行，然后停止原 Agent，备份脚本和配置，保留 `.training-state/`。升级时保留该节点现有的 `center_url`、`node_id`、`token`、目录和其他私有设置；替换脚本后通过 `--check-config` 核对解析结果。只有原配置确实存在目录不匹配时，才按目标服务器的实际布局调整路径；示例配置中的中文说明可以按需合并，但不是启用 2.0.0 GPU 并行或 2.0.1 旧 Attempt 清理的必填项。不要用开发版、使用版或另一节点的配置覆盖当前节点。

从平台「算力资源」接入窗口下载最新脚本，或从远程可访问的中心地址下载到临时文件：

```bash
cd /path/to/agent  # 替换为实际 Agent 目录
cp -p monitor_agent.py monitor_agent.py.backup
cp -p agent-config.json agent-config.json.backup
curl --fail --show-error --connect-timeout 5 --max-time 30 \
  http://YOUR_CENTER/api/compute/agent/download -o monitor_agent.new.py
python3 monitor_agent.new.py --version
```

确认输出为 `2.0.1`，核对现有配置和输出目录后，再替换并验证连接：

```bash
mv monitor_agent.new.py monitor_agent.py
chmod 600 agent-config.json
python3 monitor_agent.py --config agent-config.json --check-config
python3 monitor_agent.py --config agent-config.json --once
python3 monitor_agent.py --config agent-config.json
```

若使用 systemd，验证成功后启动对应服务即可，不要同时启动第二个前台 Agent。长期训练建议使用下方 systemd 示例，避免关闭终端结束 Agent。

如经过网关，需按 Agent 接口保留方法、请求体和 Bearer 头；除监控、训练领取、状态更新和停止查询外，1.3.0+ 还需要放行 `POST .../agent/runs/<run_id>/artifacts`、`POST .../agent/operations/poll` 以及 `POST .../agent/operations/<operation_id>/result` 和 `/chunk`，1.6.0 还需要放行 `POST .../agent/runs/<run_id>/confirm`。具体中转部署见对应工具的独立说明。

`nvidia_smi_bin` 可填写远程服务器自己的可执行文件绝对路径。运行用户需要能查询 GPU 和 `/proc/meminfo`，不需要 root。中心平台响应会更新上报间隔，因此在页面调整间隔后通常无需重启 Agent。禁用节点会拒绝该节点的所有 Agent 认证接口，包括训练状态、停止查询和文件操作；它不会直接终止远程进程，还可能阻断停止指令和最终结果回传。需要停止训练时使用任务停止操作，并等待确认后再禁用节点。更换 Token 必须更新配置并重启 Agent。

## 开机运行示例

假设程序和配置放在 `/opt/yolo-monitor-agent/`，创建 systemd 服务（替换 `YOUR_USER` 和 Python 路径为实际用户及环境）：

```ini
[Unit]
Description=YOLO Forge Compute Agent
After=network-online.target
Wants=network-online.target

[Service]
User=YOUR_USER
WorkingDirectory=/opt/yolo-monitor-agent
ExecStart=/usr/bin/python3 /opt/yolo-monitor-agent/monitor_agent.py --config /opt/yolo-monitor-agent/agent-config.json
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

配置文件应属于该用户且权限为 600。将服务保存为 `/etc/systemd/system/yolo-monitor-agent.service` 后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-monitor-agent
sudo journalctl -u yolo-monitor-agent -f
```

## 网络与权限

远程服务器主动连接中心平台，无需开放远程入站端口或提供 SSH 密码。每个 Token 限定对应节点的监控、训练和文件操作接口，数据库只存哈希；禁用或轮换 Token 可阻止旧 Agent 接入。移除节点前必须先清理其关联任务，避免丢失远程任务的管理入口。Agent 不跟随 HTTP 重定向，避免凭据转发到其他主机。

跨公网使用 HTTPS，默认验证证书。中心平台现有管理接口没有登录系统，生产部署需通过已有可信网络或认证网关限制浏览器管理入口；否则能访问管理 API 的人就能排队远程训练任务。Agent Token 不提供管理权限。网关需按上文列出的监控及训练 Agent 路径放行并保留认证头，由平台验证 Bearer Token。

超过 `max(30, 上报间隔 × 3)` 秒没有被中心接收到新上报会显示离线，判定使用中心写入的 `last_seen_at`，不使用远程机器的采样时钟。最后一次数据仍保留并标注过期；无 GPU 不等于服务器离线。算力进程只会与同一 `execution_node_id` 下的同号 PID 关联，不同节点的 PID 不会串到本机或其他远程任务。

## 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| 添加后一直“等待接入” | 填写 IP 只是注册资料，还需要在对应服务器启动 Agent；确认 `center_url` 可从远程服务器访问 |
| 配置无效或权限错误 | 使用平台下载的专属 JSON 配置；在 Linux 上执行 `chmod 600 agent-config.json`，确认当前运行用户可读取文件 |
| HTTP 401 | Token 与 `node_id` 不匹配、Token 已重置或节点已移除；核对对应节点的配置，不要复用其他服务器的 Token |
| HTTP 403 | 节点监控已禁用；在中心页面重新启用，Agent 会自动重试 |
| HTTP 409 | 采样时间早于已保存的采样；确认没有多个 Agent 共用同一节点配置，并同步服务器时间 |
| HTTP 422 | 上报格式或服务器时钟不符合要求；检查 Agent 版本并校准系统时间，时间超前超过 5 分钟会被拒绝 |
| HTTP 404 或下载不到 Agent | 确认中心地址和网关路由正确；更新平台代码后重启后端，加载新的监控接口 |
| HTTP 301 / 302 等重定向 | Agent 不跟随重定向；直接填写最终的 HTTP(S) 地址，例如网关要求 HTTPS 时使用 HTTPS 地址 |
| 网络、证书错误或超时 | 检查远程到中心的网络、中心监听端口、网关及 HTTPS 证书；Agent 保持运行后会退避重试 |
| 节点在线但没有 GPU | 使用同一个运行用户执行 `nvidia-smi`；检查驱动、执行权限和 `nvidia_smi_bin`，CPU 主机仍可正常在线 |
| 节点显示离线 | 查看 Agent 是否退出、网络是否中断及双方时钟是否一致；页面保留的旧采样不代表当前负载 |
| 关闭终端后停止更新 | 前台 Agent 已退出；使用上面的 systemd 服务配置实现长期运行 |
| 节点在线但任务一直“启动中” | 确认 Agent 达到最低版本 1.8.0（当前为 2.0.1）；异卡并行还要看到 `training.gpu_parallelism=true`。再检查 `training.ready`、中心 GPU 预约、Agent 日志与 `.training-state/active/` |
| 旧 Attempt 单独删除提示 Agent 版本过低 | 旧版平铺 Attempt 001 需要 Agent 2.0.1 才能精细清理；替换脚本并重启 Agent，确认页面报告 2.0.1 后重试 |
| 训练输出目录被拒绝 | 比较 `--check-config` 的解析结果与任务输出路径，分别核对访问目录和 `training_output_roots`；两者均通过才能训练，修改本机待复制配置不会自动更新远端进程 |
| 任务不领取、锁一直被占用 | 检查活动训练进程、中心 GPU 预约、`training_lock_file`、`${XDG_STATE_HOME:-~/.local/state}/yolo-forge-agent/` 下的 GPU / 输出根锁和各 Agent 的 `training_output_roots`；锁文件存在本身不表示锁仍被持有 |
| 永久删除提示共享 workspace 正在使用 | 查看同一系统用户下其他 1.8.0+ Agent 的节点和输出根配置；停止或调整重叠的存活 Agent 后重试，不要手工删除登记来绕过保护 |
| 远程任务长时间“正在停止” | 先到远程服务器确认当前 Attempt 的 YOLO 进程已结束；停止超过 30 秒后才可在训练画布使用“解除占用”，该操作不会终止远程进程 |
| 页面参数与远程命令不同 | 对照固定 `TRAINING_KEYS`、标量限制和远程 YOLO 版本，再检查实际命令与 `args.yaml` |
| 产物扫描失败 | 检查允许目录、链接路径、文件数是否超过 1024、单文件元数据是否超过 8 GiB |
| 禁用后停止或最终回传不生效 | 禁用会阻断 Agent 接口；恢复节点接入后确认状态，停止操作不应由禁用代替 |

每台服务器应使用各自的节点配置，同一节点只运行一个采集 Agent。Token 重置后，需要替换配置并重启 Agent；移除节点不会自动停止该服务器上的 Agent，应自行停止不再需要的监控进程。

部署在 Linux 上时，可先用实际 Python 解释器检查版本，并用当前用户查询 GPU：

```bash
python3 --version
nvidia-smi
python3 monitor_agent.py --config agent-config.json --check-config
python3 monitor_agent.py --config agent-config.json --once
```

如果中心服务器自身也要管理模型训练，中心后端仍按项目说明使用 `yolo` 环境；远程仅监控时不需要 YOLO 训练环境。Agent 配置中的地址和工具路径需要按各服务器实际环境填写。
