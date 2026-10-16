# 算力监控与可选远程训练 Agent

中心平台的「算力资源 → 添加服务器」填写名称、地址和备注，保存后下载 `monitor_agent.py` 与专属 `agent-config.json`。地址字段用于展示，不是 SSH 连接地址；中心地址需能从训练服务器访问。

Agent 默认仅采集 GPU、计算进程和主机信息。只有在远程配置显式启用训练并设置允许目录后，才会从中台领取训练任务、运行配置好的 YOLO CLI 并回报指标与日志。Agent 使用 Python 3.9+ 标准库，无第三方 Python 依赖，但当前实现依赖 Linux/POSIX 的 `fcntl`、进程组信号和 `/proc`，不应当作 Windows 通用脚本。训练所需依赖由远程 `yolo` 环境提供。GPU 显存单位为 MB，主机内存为字节。

页面操作见 [Web 说明](../README.md)，中心安装见 [项目首页](../../README.md)，双实例和备份见 [部署说明](../deploy/README.md)，已验证的问题见 [项目分析](../../docs/project-analysis.md)。

## 验证与启动

把两个下载文件放在远程服务器的同一目录，然后执行：

```bash
chmod 600 agent-config.json
python3 monitor_agent.py --config agent-config.json --once
python3 monitor_agent.py --config agent-config.json
```

`--once` 只执行一次算力采样与上报，不领取或验证训练任务；成功返回 0，失败返回非零退出码。持续运行时连接失败会退避重试；网络恢复后上报新采样。Agent 不会补传过期算力数据。终端关闭会结束前台 Agent，长期使用建议配置 systemd。

配置示例（真实 Token 只在创建或重置时提供一次）：

```json
{
  "center_url": "http://192.168.170.105:18088",
  "node_id": "平台生成的节点ID",
  "token": "平台生成的独立Token",
  "report_interval_seconds": 5,
  "nvidia_smi_bin": "nvidia-smi",
  "enable_training": false,
  "allowed_training_roots": [],
  "yolo_bin": "/data/conda/envs/yolo/bin/yolo",
  "training_cwd": "/data/yolo/yolo",
  "training_lock_file": "/data/yolo/yolo/.yolo-forge-training.lock"
}
```

## 启用远程训练

推荐使用 **Agent 1.6.0 和配套新版中台**，且 `center_url` 能从远程机器访问，再编辑远程配置文件。当前中心按 Agent 上报的版本检查能力：

| 能力 | 最低 Agent 版本 | 补充条件 |
| --- | --- | --- |
| 基础远程训练、状态、指标与日志尾部 | 1.2.0 | 显式启用训练，设置环境和允许目录 |
| 远程断点续训、产物扫描传输及永久删除 | 1.3.0 | 检查点或目标文件可用，节点保持接入 |
| 多 Agent 共享训练锁 | 1.4.0 | 必须使用同一锁文件，不能仅凭版本保证互斥 |
| 远程 YOLO 训练过程解析 | 1.5.0 | 仅目标检测任务；按批次采样真实输入、层统计和损失摘要 |
| 训练领取租约与启动确认 | 1.6.0 | 未确认领取在 90 秒后可重新分发，降低响应丢失导致的长期“启动中” |

低于 1.2.0 的旧 Agent 不能领取新训练任务；1.2.x 可以进行基础训练，但不具备完整的续训和文件管理能力。1.2.0～1.5.x 领取训练时没有 1.6.0 的租约保护，因此完整远程功能应统一升级到 1.6.0。配置示例：

```json
{
  "enable_training": true,
  "allowed_training_roots": ["/data/yolo/yolo", "/data/yolo/datasets", "/data/yolo/runs"],
  "yolo_bin": "/data/conda/envs/yolo/bin/yolo",
  "training_cwd": "/data/yolo/yolo",
  "training_lock_file": "/data/yolo/yolo/.yolo-forge-training.lock",
  "enable_training_trace": true
}
```

这些字段须加到现有 JSON 对象中，保留原来的 `center_url`、`node_id`、`token` 等字段。示例路径需要与执行服务器实际的模型、数据目录和 Conda 环境一致。`allowed_training_roots` 应仅列模型文件、数据 YAML、输出根目录所在的最小目录；工作目录和输出根目录须事先存在，运行 Agent 的用户须有相应写权限。训练产物位于 `<输出根目录>/<任务类型>/<任务 ID>/attempt-001`。启动前 Agent 检查模型文件和数据 YAML 是否存在、路径是否落在允许目录内；它不会逐一检查 YAML 引用的全部图片路径是否也在允许目录内。命令通过参数数组执行，不执行任意 Shell。建议用具有训练目录写入权的专用非 root 用户运行 Agent。

`training_lock_file` 可省略，默认位于 `training_cwd/.yolo-forge-training.lock`。同一服务器若同时运行开发版和使用版的两个 Agent，应让它们使用相同的 `training_cwd` 或显式配置同一个锁文件，推荐使用同一系统账户。Agent 尝试将锁文件权限设为 `660`；不同账户共用时还需核对文件所有权、组、父目录访问权限及修改权限。Agent 只在持有操作系统文件锁时领取任务，进程退出会自动释放；锁文件本身保留不代表仍被占用。两套平台仍需各自的节点 ID、Token 和 Agent 目录。

锁只约束访问同一锁文件的 Agent，不约束中心本机调度器、不同锁路径下的 Agent 或平台外启动的 YOLO 进程。共用锁文件会串行化整项训练，即使两个任务选择不同 GPU，也不会并行领取；它不是按 GPU 分配的全主机资源调度器。

在新建任务弹窗选择已注册的远程服务器，再输入远程机器上真实的模型文件、数据集 YAML 和输出根目录的绝对路径。数据集须已在远程服务器可用；中台不会上传或复制它们。任务按节点隔离排队，同一节点一次领取一项训练；Agent 在线时主动领取；可在平台请求停止，Agent 收到指令后终止对应进程。指标和最近 100 行日志回传到中台，完整日志与权重保存在远程机器。

Agent 1.5.0+ 会在远程目标检测训练中自动采集最多 24 个真实 Batch 的输入张量、网络层统计、损失和采样图片，并增量上传到中台的过程解析接口。平台不会上传完整特征图、模型权重或数据集；采样图片单张最多 512 KiB。采集失败不会中断训练，网络暂时不可用时会在下一次轮询重试。若远程服务器资源紧张，可在配置中设置 `"enable_training_trace": false`，关闭过程采集但保留普通训练、指标和日志回传。替换 Agent 后需重新领取或重新启动训练，已经运行的旧 Attempt 不会补采历史 Batch。

中心从源码配置动态生成参数目录，Agent 1.6.0 的 `TRAINING_KEYS` 是固定集合。例如中心已有 `cls_remap`，Agent 的预设白名单尚未包含，作为预设参数时不会加入远程命令。Agent 参数序列化只接受标量，列表和字典可能被拒绝；不能假定本机与远程的参数完全等价。配置训练后应核对实际命令与产物中的 `args.yaml`，并确认远程 YOLO 版本支持所选参数。

### 远程任务与文件操作（Agent 1.3.0）

远程任务使用与本机任务相同的重命名、克隆、重新训练、回收站和历史运行入口。克隆保留执行服务器和远程路径，不复制数据集或权重。

- **断点续训**：任务须已停止或中断，Agent 已同步该运行的 `weights/last.pt`。续训生成新 Attempt，保留旧运行的文件和指标；使用检查点内的优化器与训练状态。总轮数和数据集必须保持与上次运行一致；想改总轮数或数据集时使用重新训练。
- **产物列表、图片和下载**：Agent 周期上报文件元数据，打开历史运行会排队扫描。预览或下载时才从对应服务器分块传输到中台缓存，校验大小和 SHA256 后提供文件。断网时显示准备中或重试，不会读取中台上的同名远程路径。中心接受的单文件元数据大小上限为 8 GiB；扫描发现超过 1024 个文件会报错，不是只同步前 1024 个。缓存位于中台 `PLATFORM_DATA_ROOT/remote-artifacts/`，永久删除任务时一并清理。
- **移入回收站和恢复**：仅隐藏或恢复平台记录，不立即删除远程文件。活动任务须先停止。
- **永久删除**：提交后显示“远程清理中”，由 Agent 清理该任务各次 Attempt 和对应独立日志，成功确认后中台才删除任务记录。离线时保留记录等待连接；失败显示原因，允许重试清理。提交永久删除后不能恢复，包括已经部分清理的失败任务。

文件操作仅限配置允许目录内、精确匹配任务及 Attempt 的路径，拒绝路径逃逸及链接目录；清理前核对全部目标，保护训练项目、YOLO 程序以及平台已登记的模型和数据 YAML 输入。没有向平台登记的外部文件引用不在这份保护清单内，应避免把共享训练输入放在任务输出目录中。文件操作通过独立工作线程执行，不阻塞算力心跳或停止指令查询。

### 日志、断网与停止原因

完整日志独立保存到 `<输出根目录>/.platform-logs/YYYYMMDD/<任务 ID>/attempt-NNN.log`，日期由中台下发。例如：

```text
/data/yolo/yolo/runs/.platform-logs/20261012/b8a5698c8a2c4a6d/attempt-004.log
```

权重、`results.csv`、训练图片仍在原来的 Attempt 产物目录。独立日志不会被 YOLO 双 GPU 的 DDP 启动目录清理删除。新 Attempt 禁止覆盖已有产物或日志；旧 Attempt 路径保持原样，已删除的旧日志无法恢复。

训练进程启动后，即使首次状态上报超时或后续网络中断，Agent 仍保持训练运行。网络恢复后重传最近日志和未确认指标；超过 256 个 Epoch 的积压指标分批补传，中台按运行 ID 和 Epoch 去重。日志尾部是最近 100 行，完整输出需在远程日志文件中查看。

页面日志窗口的“复制全部”和“下载日志”都只处理已回传的尾部，不能取回完整远程日志。独立 `.platform-logs` 目录在 Attempt 产物目录之外，普通产物扫描也不会把这些日志作为训练产物上传。完整输出应在执行服务器按上述路径获取；两个按钮的已知问题见 [日志查看与导出](../README.md#日志查看与导出)。

Agent 1.6.0 领取任务时使用随机租约，并在启动 YOLO 前向中心确认。若领取响应没有到达 Agent，中心会在 90 秒租约过期后重新分发；过期租约不能继续更新运行，避免旧响应导致重复接管。1.2.0～1.5.x 为兼容模式，没有训练领取租约；中心仍允许它们领取相应版本支持的任务，但不具备这一恢复保护。

训练结束时先将最终状态、日志尾部及未确认指标写到 **Agent 配置文件旁的 `.training-state/<运行 ID>.json`**，收到中台确认后才删除。Agent 重启后也会重试这些文件，补传完成前不领取下一项训练。此目录不含 Token，但含训练日志，应保持私有权限，并在升级时保留；认证或路径错误需修正后才能补传。

平台停止指令和 Agent 退出会分别显示对应原因，不再统一标记为用户停止。平台停止事件记录请求时间、运行 ID 和连接来源 IP；经网关访问时可能记录网关地址，来源 IP 不能直接代表具体操作者。网络失联不自动判定训练失败或停止。

**Agent 正常退出或服务重启仍会终止正在运行的训练。** 本版本只恢复已落盘的最终结果，不接管因断电、强制结束 Agent 而遗留的训练进程，也不自动恢复训练；升级前应确认没有运行中的训练，避免重复执行。

### 更新 Agent

先确认没有训练运行，然后停止原 Agent，备份并只替换 `monitor_agent.py`；保留 `agent-config.json` 与 `.training-state/`。从平台「算力资源」接入窗口下载最新脚本，或从远程可访问的中心地址下载到临时文件：

```bash
cd /data/yolo/Agent
cp -p monitor_agent.py monitor_agent.py.backup
curl --fail --show-error --connect-timeout 5 --max-time 30 \
  http://YOUR_CENTER/api/compute/agent/download -o monitor_agent.new.py
python3 monitor_agent.new.py --version
```

确认输出为 `1.6.0` 后，再替换并验证连接：

```bash
mv monitor_agent.new.py monitor_agent.py
chmod 600 agent-config.json
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

超过 `max(30, 上报间隔 × 3)` 秒没有新采样会显示离线，最后一次数据仍保留并标注过期；无 GPU 不等于服务器离线。远程 PID 不与本机训练进程关联，避免同号 PID 混淆。

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
| 节点在线但任务一直“启动中” | 先确认 Agent 为 1.6.0、服务器时钟正常并检查 Agent 日志；1.6.0 未确认领取会在 90 秒后重新分发，旧 Agent 没有该保护 |
| 任务不领取、锁一直被占用 | 检查其他共用锁文件的 Agent 和锁文件权限；文件存在本身不表示锁仍被持有 |
| 页面参数与远程命令不同 | 对照固定 `TRAINING_KEYS`、标量限制和远程 YOLO 版本，再检查实际命令与 `args.yaml` |
| 产物扫描失败 | 检查允许目录、链接路径、文件数是否超过 1024、单文件元数据是否超过 8 GiB |
| 禁用后停止或最终回传不生效 | 禁用会阻断 Agent 接口；恢复节点接入后确认状态，停止操作不应由禁用代替 |

每台服务器应使用各自的节点配置，同一节点只运行一个采集 Agent。Token 重置后，需要替换配置并重启 Agent；移除节点不会自动停止该服务器上的 Agent，应自行停止不再需要的监控进程。

部署在 Linux 上时，可先用实际 Python 解释器检查版本，并用当前用户查询 GPU：

```bash
python3 --version
nvidia-smi
python3 monitor_agent.py --config agent-config.json --once
```

如果中心服务器自身也要管理模型训练，中心后端仍按项目说明使用 `yolo` 环境；远程仅监控时不需要 YOLO 训练环境。Agent 配置中的地址和工具路径需要按各服务器实际环境填写。
