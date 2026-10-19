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

`--check-config` 只在本机检查配置、YOLO 程序、输出写入、状态目录和共享锁，不连接中心、不打印 Token。`--once` 执行一次算力采样与上报，不领取训练任务；成功返回 0，失败返回非零退出码。持续运行时连接失败会退避重试；网络恢复后上报新采样。Agent 不会补传过期算力数据。终端关闭会结束前台 Agent，长期使用建议配置 systemd。

配置示例（真实 Token 只在创建或重置时提供一次）：

```json
{
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

使用 **Agent 1.8.0 和配套的平台 1.1.0**，并确保 `center_url` 能从远程机器访问。当前中心将 1.8.0 作为新远程训练的最低版本；以下表格保留协议能力的版本演进，不表示旧 Agent 仍可向当前中心领取训练：

| 能力 | 最低 Agent 版本 | 补充条件 |
| --- | --- | --- |
| 基础远程训练、状态、指标与日志尾部 | 1.2.0 | 显式启用训练，设置环境和允许目录 |
| 远程断点续训、产物扫描传输及永久删除 | 1.3.0 | 检查点或目标文件可用，节点保持接入 |
| 多 Agent 共享训练锁 | 1.4.0 | 必须使用同一锁文件，不能仅凭版本保证互斥 |
| 远程 YOLO 训练过程解析 | 1.5.0 | 仅目标检测任务；按批次采样真实输入、层统计和损失摘要 |
| 训练领取租约与启动确认 | 1.6.0 | 未确认领取在 90 秒后可重新分发，降低响应丢失导致的长期“启动中” |
| 确认重试、活动运行恢复与单实例 | 1.7.0 | 配合新版中心处理确认响应丢失、确认前停止和 Agent 异常退出 |
| 训练就绪门禁、层级输出根锁、共享根删除保护和恢复结果判定 | 1.8.0 | 中心按版本、`last_seen_at` 和 `training.ready` 控制训练入口；同机多 Agent 必须使用同一专用系统账户才能共享完整 workspace 保护 |

当前中心在创建、启动、续训、重新训练和领取阶段都会拒绝低于 1.8.0 的 Agent。只需监控的旧 Agent 仍可上报算力，但不应用于新的远程训练。配置示例中的 `/srv/...` 是占位路径，必须按远程服务器实际目录替换：

```json
{
  "enable_training": true,
  "allowed_training_roots": ["/srv/yolo/models", "/srv/yolo/datasets", "/srv/yolo/runs"],
  "training_output_roots": ["/srv/yolo/runs"],
  "yolo_bin": "yolo",
  "enable_training_trace": true
}
```

这些字段须加到现有 JSON 对象中，保留原来的 `center_url`、`node_id`、`token` 等字段。`yolo_bin` 默认为 `yolo`，由 Agent 运行环境的 `PATH` 解析；需要锁定特定环境时再填写该服务器上的可执行文件绝对路径。`training_cwd` 可省略，默认为 `agent-config.json` 所在目录；需要在特定项目目录启动 YOLO 时再显式配置。`allowed_training_roots` 应仅列模型文件、数据 YAML、输出根目录所在的最小目录；输出根目录须事先存在，运行 Agent 的用户须有相应写权限。训练产物位于 `<输出根目录>/<任务类型>/<任务 ID>/attempt-001`。启动前 Agent 检查模型文件和数据 YAML 是否存在、路径是否落在允许目录内；它不会逐一检查 YAML 引用的全部图片路径是否也在允许目录内。命令通过参数数组执行，不执行任意 Shell。建议用具有训练目录写入权的专用非 root 用户运行 Agent。

`training_output_roots` 明确哪些允许目录可作为训练输出，每一项必须位于 `allowed_training_roots` 内。旧配置未填写时 Agent 会兼容运行，但只能判断允许目录中是否至少有一个可写；每次启动仍会对任务实际输出根目录再做写入检查。

`training_lock_file` 仍可选，默认位于 `training_cwd/.yolo-forge-training.lock`，用于保持单个 workspace 的兼容互斥；锁文件由 YOLO 子进程继承，Agent 意外结束时不会在训练存活期间提前释放。Agent 1.8.0 另外在 `${XDG_STATE_HOME:-~/.local/state}/yolo-forge-agent/` 中对 `training_output_roots` 建立层级意向锁：相同根目录或父子根目录必然互斥，不相交的兄弟根目录可以并行。因此同一系统用户下的两个 1.8.0 Agent 不需要为了输出目录保护而手工填成同一个 `training_lock_file`。若为兼容旧部署而显式共用该文件，它仍是全局互斥锁，会让不相交的兄弟输出根也串行。

workspace 登记以 `node_id` 的稳定哈希作为 owner，不依赖配置文件路径或进程 PID。永久删除前，Agent 会在持有层级锁的情况下检查共享根目录的其他存活 owner；只要另一个 Agent 仍声明与删除目标重叠的输出根，就拒绝删除并返回可重试错误。这套协调使用权限为 `0700` 的用户私有状态目录，因此同机多 Agent 必须使用同一专用系统账户才能共享完整的层级锁和删除 owner 保护。不同账户即使显式共用 `training_lock_file`，也只获得较粗的训练互斥，无法共享 workspace 删除登记。这些锁不约束中心本机调度器或平台外启动的 YOLO 进程，也不是按 GPU 分配的全主机资源调度器。两套平台仍需各自的节点 ID、Token 和 Agent 配置。

在新建任务弹窗选择已注册的远程服务器，再输入远程机器上真实的模型文件、数据集 YAML 和输出根目录的绝对路径。数据集须已在远程服务器可用；中台不会上传或复制它们。中心以自己记录的 `last_seen_at` 判定在线，不使用远程采样时钟作为连通性依据；同时要求 Agent 版本至少为 1.8.0，且最近上报中 `training.enabled=true` 与 `training.ready=true`。任一条件不满足时，创建、启动、续训、重新训练和领取都会返回具体原因；Agent 自身也会暂停领取。任务按节点隔离排队，同一节点一次领取一项训练。指标和最近 100 行日志回传到中台，完整日志与权重保存在远程机器。

Agent 1.5.0+ 会在远程目标检测训练中自动采集最多 24 个真实 Batch 的输入张量、网络层统计、损失和采样图片，并增量上传到中台的过程解析接口。平台不会上传完整特征图、模型权重或数据集；采样图片单张最多 512 KiB。采集失败不会中断训练，网络暂时不可用时会在下一次轮询重试。若远程服务器资源紧张，可在配置中设置 `"enable_training_trace": false`，关闭过程采集但保留普通训练、指标和日志回传。替换 Agent 后需重新领取或重新启动训练，已经运行的旧 Attempt 不会补采历史 Batch。

中心从源码配置动态生成参数目录，Agent 1.8.0 与平台 1.1.0 的发布目录一致。Agent 参数序列化只接受标量，列表和字典会被拒绝；后续中心目录变化时仍需同步 Agent。配置训练后应核对实际命令与产物中的 `args.yaml`，并确认远程 YOLO 版本支持所选参数。

### 远程任务与文件操作（Agent 1.3.0）

远程任务使用与本机任务相同的重命名、克隆、重新训练、回收站和历史运行入口。克隆保留执行服务器和远程路径，不复制数据集或权重。

- **断点续训**：任务须已停止或中断，Agent 已同步该运行的 `weights/last.pt`。续训生成新 Attempt，保留旧运行的文件和指标；使用检查点内的优化器与训练状态。总轮数和数据集必须保持与上次运行一致；想改总轮数或数据集时使用重新训练。
- **产物列表、图片和下载**：Agent 周期上报文件元数据，打开历史运行会排队扫描。预览或下载时才从对应服务器分块传输到中台缓存，校验大小和 SHA256 后提供文件。断网时显示准备中或重试，不会读取中台上的同名远程路径。中心接受的单文件元数据大小上限为 8 GiB；扫描发现超过 1024 个文件会报错，不是只同步前 1024 个。缓存位于中台 `PLATFORM_DATA_ROOT/remote-artifacts/`，永久删除任务时一并清理。
- **移入回收站和恢复**：仅隐藏或恢复平台记录，不立即删除远程文件。活动任务须先停止。
- **永久删除**：提交后显示“远程清理中”，由 Agent 清理该任务各次 Attempt 和对应独立日志，成功确认后中台才删除任务记录。离线时保留记录等待连接；失败显示原因，允许重试清理。提交永久删除后不能恢复，包括已经部分清理的失败任务。Attempt 记录的输出根目录不存在、允许目录失效或磁盘挂载离线时，Agent 会返回可重试失败，不会把“无法核验”当成“已删除”。

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

训练结束时先将最终状态、日志尾部及未确认指标写到 **Agent 配置文件旁的 `.training-state/<运行 ID>.json`**，收到中台确认后才删除。Agent 重启后也会重试这些文件，补传完成前不领取下一项训练。此目录不含 Token，但含训练日志，应保持私有权限，并在升级时保留；认证或路径错误需修正后才能补传。

平台停止指令和 Agent 退出会分别显示对应原因，不再统一标记为用户停止。平台停止事件记录请求时间、运行 ID 和连接来源 IP；经网关访问时可能记录网关地址，来源 IP 不能直接代表具体操作者。网络失联不自动判定训练失败或停止。

Agent 将活动任务、启动命令、PID、Linux 进程启动时刻、命令行哈希和 boot ID 写入 `.training-state/active/`，不写入 Token。强制结束 Agent 后训练进程仍存活时，新 Agent 只在进程身份完全匹配时接管进度和停止指令，不启动重复训练。接管的进程自然结束后，Agent 1.8.0 使用已完成 Epoch、`best.pt` / `last.pt` 和日志完成标记综合判定：证据足够时回传 `completed`，无法确认时回传 `interrupted` 并保留产物供核对，不再一律标记为失败。正常收到 SIGINT/SIGTERM 时 Agent 仍会先停止它管理的训练并回传状态。

远程任务停止后长时间保持 `stopping` 时，先登录对应训练服务器，根据 Attempt、命令和 PID 确认该 YOLO 进程已经结束。停止状态持续至少 30 秒后，当前训练画布才显示“解除占用”；再按页面提示明确确认远程进程已停止。该操作只将当前 Attempt 标记为 `interrupted` 并解除中心的节点占用，**不会向远程进程发送终止信号**。若旧进程仍在运行就解除占用，新任务可能与它同时占用 GPU。当前 Attempt 已变化或停止状态刚刚更新时，中心会拒绝这次操作，应刷新后重新核对。

每台远程服务器可以配置不同的 `training_cwd`、`allowed_training_roots` 和 `training_output_roots`，平台不会把某一台服务器的绝对路径当作所有节点的固定路径。`training_cwd` 未设置时取配置文件所在目录，不依赖某台服务器的预设项目路径。训练、产物扫描和下载始终按当前 Agent 的允许目录严格校验。永久删除时，不存在的 Attempt 只有在其输出根目录仍位于当前可用的允许目录内时才视为已清理；配置外路径、允许目录消失或挂载离线都会返回可重试错误，不删除中心记录。

### 更新 Agent

先确认没有训练运行，然后停止原 Agent，备份并只替换 `monitor_agent.py`；保留 `agent-config.json` 与 `.training-state/`。从平台「算力资源」接入窗口下载最新脚本，或从远程可访问的中心地址下载到临时文件：

```bash
cd /path/to/agent  # 替换为实际 Agent 目录
cp -p monitor_agent.py monitor_agent.py.backup
curl --fail --show-error --connect-timeout 5 --max-time 30 \
  http://YOUR_CENTER/api/compute/agent/download -o monitor_agent.new.py
python3 monitor_agent.new.py --version
```

确认输出为 `1.8.0` 后，再替换并验证连接：

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
| 节点在线但任务一直“启动中” | 确认中心与 Agent 均为配套的 1.8.0，在算力页查看 `training.ready` 及原因，再检查 Agent 日志与 `.training-state/active/` |
| 任务不领取、锁一直被占用 | 检查活动训练进程、`training_lock_file`、`${XDG_STATE_HOME:-~/.local/state}/yolo-forge-agent/locks/` 和各 Agent 的 `training_output_roots`；锁文件存在本身不表示锁仍被持有 |
| 永久删除提示共享 workspace 正在使用 | 查看同一系统用户下其他 1.8.0 Agent 的节点和输出根配置；停止或调整重叠的存活 Agent 后重试，不要手工删除登记来绕过保护 |
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
