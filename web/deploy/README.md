# 双实例部署与维护

## 1.7.0 Agent 保留训练重载与进度接管

配套 Agent **2.3.0**：默认退出保留已启动的训练进程、原 Run/Attempt、日志和资源锁；再次启动核验原进程后接管指标、产物与训练采样。支持 `SIGHUP` 原进程重载，systemd 使用 `systemctl reload`，不要在训练中用默认控制组配置的 `systemctl restart/stop`。需要退出时同时停止训练，启动 Agent 时指定 `--shutdown-policy stop`；平台停止按钮仍始终有效。新 Agent 在监控上报之前恢复活动任务，采样上报失败不阻断已有训练的进度回传。

训练画布显示远程 Agent 待重连提示并保留最后进度，重连后自动恢复；本机训练及历史结束状态保持原语义。沿用历史活动记录和原节点配置，不迁移运行表、不新增必填路径。首次从 2.2.x 及更旧版本升级应等待训练空闲；旧进程不会因覆盖脚本获得新退出行为。详情见 Agent 说明的“保留训练重启与重载”。


## 1.6.0 已有数据集扫描与批量登记

数据集管理新增后台目录扫描、候选分页筛选、跨页多选和确认后批量登记；支持本机与远程、嵌套目录、重叠位置去重、进度恢复、取消及逐项重试。初检只读取目录和小型 YAML，正式登记沿用完整图片、标签与 SHA-256 校验；默认外部登记，不修改原数据。远程扫描需要 Agent **2.2.0** 并上报 `training.dataset_discovery=true`，旧版手动登记和训练协议保留。升级后需重启开发版后端，数据库自动创建扫描与批次相关表。


## 1.5.3 数据集登记与迁移弹窗布局

登记和复制／迁移弹窗改用单列布局，分别选择服务器与保存位置，完整路径独立显示并支持长路径换行。说明、文件管理权限与确认操作分区呈现，窄屏及低窗口可滚动操作。沿用已有位置编号和提交接口，兼容历史数据及本地、远程服务器；无需升级 Agent。


## 1.5.2 数据集操作记录清理

操作记录支持单条删除、本页多选删除，以及按服务器、类型、状态、名称和日期范围批量清理。仅已完成或已取消的记录可移除，提交前预览准确数量；候选记录变化时需重新确认。删除只影响历史展示和统计，保留数据集、实际文件、上传结果、远程回执与临时文件回收逻辑。旧记录直接兼容，Agent 继续使用 2.1.0，无需更新；开发版后端重启后自动创建历史移除标记表。


## 1.5.1 数据集管理交互优化

数据集页面通过“保存位置”弹窗和“操作记录”侧边面板管理配置与历史，主列表保留筛选和滚动位置。操作记录支持全部历史分页、状态/类型/服务器/名称筛选，默认优先显示进行中和失败操作；详情按需展开，新增记录提示后手动查看。服务器选择器支持搜索、键盘操作及本机/离线远程节点；“可删除”筛选同时校验文件管理权、引用和未完成操作。本次无需升级 Agent（继续使用 2.1.0），更新后由维护者安排开发版后端重启。


## 1.5.0 数据集存储与跨节点管理

新增本机/远程具名保存位置、默认目录与真实路径/空间检查；上传目标固定并支持 SHA-256 断点恢复，后台传输、登记已有目录、跨节点复制迁移、外部目录接管及引用保护删除。历史上传目录自动兼容，迁移保留编号与原路径历史，操作可追踪、重试和提交前取消。远程新能力需要 Agent **2.1.0**，原有训练最低兼容门槛保持不变。完整配置、空间要求、恢复和删除规则见 [数据集存储说明](../../docs/dataset-storage.md)。



本仓库的服务模板按两套独立 checkout 配置：开发版使用 `18088`，使用版使用 `18089`。两套服务必须使用不同的 systemd 服务名，不能都安装成 `yolo-platform.service`。模板存在不代表两套服务已在目标机器安装或运行。

安装依赖和构建步骤见 [项目首页](../../README.md)；操作规则见 [Web 说明](../README.md)；远程服务器见 [Agent 说明](../agent/README.md)；架构与故障证据见 [项目分析](../../docs/project-analysis.md)。当前开发平台版本为 **1.7.0**，配套远程 Agent 为 **2.3.0**。

仓库同时保存 `yolo-platform-dev.service` 和 `yolo-platform-prod.service`，分别配置两套路径和端口。环境差异放在各实例的 `.env` 中，服务模板也分开维护，减少因开发/使用版配置混用引起的冲突。

| 实例 | 项目目录 | 端口 | systemd 建议名称 | 默认数据库 |
| --- | --- | --- | --- | --- |
| 开发版 | `<开发版 checkout>` | `18088` | `yolo-platform-dev.service` | 本 checkout 的 `web/data/platform.db` |
| 使用版 | `<使用版 checkout>` | `18089` | `yolo-platform-prod.service` | 本 checkout 的 `web/data/platform.db` |

`YOLO_PROJECT_ROOT=.` 与相对的 `DATABASE_URL=sqlite:///web/data/platform.db` 都由后端按代码所在项目解析，不受启动终端目录影响。不要让两套 `.env` 指向同一个绝对数据库、数据目录或运行目录。

## 配置和运行账户

先安装中心后端、实际训练环境和前端依赖并构建前端。模板中的项目路径、Python 路径和端口是部署示例，必须按目标服务器调整。`YOLO_BIN` 未设置时，后端先查找当前 Python 环境同目录的 `yolo`，再查找 `PATH`；生产环境可显式指向已验证的训练程序，但不应把某台开发机路径写入通用配置。

两套实例分别准备自己的 `web/.env`。只在文件不存在时从 `.env.example` 创建，已有文件应逐项核对，不能直接覆盖。例如在开发项目根目录执行：

```bash
cd /path/to/development-checkout  # 替换为实际开发版 checkout
if [ ! -e web/.env ]; then
  cp web/.env.example web/.env
fi
chmod 600 web/.env
```

虽然模板使用 `EnvironmentFile=-...`，`ExecStartPre` 仍检查 `.env` 可读，因此文件缺失会阻止服务启动。直接运行 Uvicorn 不会自动加载该文件；上述配置加载由 systemd 完成。

| 配置项 | 默认或模板行为 | 迁移时要核对的内容 |
| --- | --- | --- |
| `YOLO_PROJECT_ROOT` | 相对代码 checkout 解析 | 两套项目各自的根目录 |
| `PLATFORM_DATA_ROOT` | `web/data` | 上传暂存、远程缓存、持久模型测试历史 `model-test-history/` 和临时测试工作目录 `model-test-work/` 等中心状态目录 |
| `PLATFORM_RUNS_ROOT`、`PLATFORM_RUNTIME_ROOT`、`PLATFORM_LOG_ROOT` | 未单独设置时跟随数据根目录 | `.env.example` 已独立设置前两项，迁移数据根目录时须一起改；日志项需按需补充 |
| `PLATFORM_MODEL_TEST_ROOTS` | 项目相对的 `pt` | 独立本机模型测试的只读边界；多个值使用操作系统路径分隔符，两个 checkout 应保持各自目录 |
| `PLATFORM_MODEL_TEST_HISTORY_QUOTA_BYTES`、`PLATFORM_MODEL_TEST_TASK_QUOTA_BYTES` | 默认 20 GiB / 5 GiB | 中心保存的模型测试历史总量和单任务量；单位为字节，设为 `0` 才关闭对应配额 |
| `PLATFORM_MODEL_TEST_MIN_FREE_BYTES`、`PLATFORM_MODEL_TEST_IMAGE_RESERVE_BYTES` | 默认 2 GiB / 64 MiB | 创建测试时保留的最低可用空间和每张图片的保守预留；单位为字节 |
| `PLATFORM_REMOTE_MODEL_CACHE_QUOTA_BYTES`、`PLATFORM_REMOTE_MODEL_CACHE_MIN_FREE_BYTES` | 默认 20 GiB / 2 GiB | 独立远程模型固定缓存的总配额和接收时最低磁盘余量；单位为字节，设为 `0` 才关闭对应保护 |
| `DATABASE_URL` | `sqlite:///web/data/platform.db` | 独立于数据根目录，修改 `PLATFORM_DATA_ROOT` 不会自动修改数据库地址 |
| `YOLO_BIN`、`ULTRALYTICS_ROOT` | 训练程序与源码配置目录 | 参数目录来自源码，不是对实际 CLI 能力的探测 |
| `PLATFORM_INSTANCE_NAME` | 模板分别设置 `development`、`production`，但 `.env` 中同名项优先 | 将示例的 `local` 改为对应实例名称，用健康接口核对最终生效值 |
| `PLATFORM_VERSION` | 默认读取 checkout 根目录 `VERSION` | 通常不覆盖；仅打包无 `VERSION` 的运行环境才需要显式设置 |

目录配置变更不会迁移旧文件，也不会自动更新数据库记录里的绝对路径。迁移须同时处理数据库、运行目录、模型、数据集和远程 Agent 的中心地址。项目根目录的 `pt/.gitkeep` 只保证新 checkout 具有默认目录，不包含任何模型权重；各实例须自行放入或挂载 `.pt` 文件。

仓库服务模板没有 `User=` / `Group=`，直接作为系统服务安装通常会以 root 运行。部署前应设置实际专用账户及组，并让其能够读取代码、`.env`、Python 环境和输入资源，写入数据库父目录及各输出目录。`UMask=0077` 和 `NoNewPrivileges=true` 不会替代账户配置，也不会修改旧文件的所有权或权限。

## 安装服务文件

准备好配置和账户后，复制并校验服务。更新前先确认没有本机或远程训练正在运行，也没有未完成的文件传输或永久删除操作。

```bash
sudo cp /path/to/development-checkout/web/deploy/yolo-platform-dev.service \
  /etc/systemd/system/yolo-platform-dev.service
sudo cp /path/to/production-checkout/web/deploy/yolo-platform-prod.service \
  /etc/systemd/system/yolo-platform-prod.service
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/yolo-platform-dev.service
sudo systemd-analyze verify /etc/systemd/system/yolo-platform-prod.service
```

需要启动时分别执行：

```bash
sudo systemctl start yolo-platform-dev
sudo systemctl start yolo-platform-prod
curl --fail http://127.0.0.1:18088/api/health
curl --fail http://127.0.0.1:18089/api/health
```

核对各自 `.env` 中的 `PLATFORM_INSTANCE_NAME` 与模板一致：开发版为 `development`，使用版为 `production`。`EnvironmentFile` 中的同名变量会覆盖服务的 `Environment=` 设置，直接复制示例中的 `local` 会使健康接口显示 `local`；自定义启动未设置时可能显示项目目录名，例如 `yolo`。服务使用单个 Uvicorn worker，且同一数据库和目录只由一个后端实例管理；不能通过增加 worker 或另开共享状态的 Uvicorn 进程扩容。

仓库中的旧 `yolo-platform.service` 对应开发路径，不适合两套实例都安装成同名服务。安装前先检查已有服务和端口监听，避免旧服务与新模板重复启动。

## 更新与备份顺序

平台 1.4.5 为 `model_assets` 增加可空的 `deleted_at` 与索引，首次启动在 SQLite 备份完成后自动迁移。已有模型登记默认保持可见，历史 ID 和关联不变。此升级无须更新 Agent，也不需要调整模型目录配置；应同时更新目标实例的后端和前端构建。移除登记只影响模型库可见性，历史仍引用的远程缓存继续需要纳入备份。

平台 1.4.8 修复服务重启接管后将正常早停误判为中断的问题，需要同时更新前后端；配套 Agent 2.0.5 支持远程运行结束记录与恢复判断。任务状态与实际轮数分开展示，不改变最后一轮指标的含义。平台 1.4.7 完善的本机和远程整任务清理继续保留，远程有产物任务的永久删除至少需要 Agent 2.0.4。升级远程脚本时保留原有节点 ID、Token、路径和 `.training-state/`，不要覆盖真实配置；更新后由维护者安排服务重启。最低训练版本与独立模型传输版本门槛保持不变。

1.4.8 为 `training_tasks` 和 `training_runs` 增加 `completion_info` 字段，首次启动会先备份 SQLite 数据库，再自动迁移；已有记录默认为空，不改写训练指标。有完整结束证据且因接管进程丢失退出码而被误标中断的本机运行，会在调度器启动时校正；人工停止和缺少结束证据的运行保持原状态。

1. 确认没有训练运行、模型测试、图片上传、清理中任务或正在传输的产物，记录当前版本、环境配置和运行目录。停止目标服务，或以其他方式确保备份期间不会再发生模型测试上传、执行和删除写入。
2. 在对应项目根目录执行数据库备份。默认数据库未改动时：

   ```bash
   python web/deploy/backup_database.py
   ```

3. 将对应 `PLATFORM_DATA_ROOT/model-test-history/` 与第 2 步数据库快照作为同一备份集复制；再单独备份需要保留的模型、数据集、运行产物、日志、运行时配置与 `.env`。远程文件和 Agent 的 `.training-state/` 在对应远程机器备份。
4. 拉取或复制代码，按 [开发检查](../README.md#开发检查与排障) 构建前端并运行测试。后端测试应使用临时数据库和数据目录，不对正式数据库执行测试。
5. 只重启需要更新的那个服务，不要同时覆盖两套 `.env` 或 `web/data/`。后端启动会执行数据库迁移，保留升级前的一致备份。
6. 检查健康接口、服务日志、首页及任务状态；确认这一实例正常后再更新另一实例。

备份脚本使用 SQLite Backup API，包含 WAL 中已提交的数据，并执行 `integrity_check`、设置文件权限为 600。默认输入为**脚本所在项目**的 `web/data/platform.db`，输出为该项目的 `web/data/backups/`；相对参数按 `web/` 解析。脚本不读取 `.env`、`DATABASE_URL` 或 `PLATFORM_DATA_ROOT`，自定义数据库必须显式指定：

```bash
# 示例：用实际数据库和备份路径替换下列绝对路径
python web/deploy/backup_database.py \
  --source /srv/yolo-state/platform.db \
  --output-dir /srv/yolo-backups
```

脚本不会自动删除旧备份，也不包含模型、数据集、产物、日志或 `model-test-history/`。SQLite 快照中的 `model_assets`、`model_tests`、`model_test_images` 是模型来源、测试索引和元数据；对应原图、结果和下载文件位于 `model-test-history/`。Backup API 不会冻结这些文件的并发变化，因此需要可完整恢复且相互匹配的模型测试历史时，必须在目标服务停止或静默期间依次生成数据库快照和复制历史目录。按上述步骤确认没有活动模型测试后，`model-test-work/` 不包含需要备份的持久结果；若需要保留已抓取的远程指定模型以便 Agent 离线后继续新建检测，还应在同一时点保留 `remote-artifacts/` 内对应缓存。缓存容量检查会回收没有当前资产或历史测试引用的旧版本，因此数据库和缓存必须取自同一静默时点。若在活动测试期间强行备份，有效的固定权重副本也必须与数据库状态保持同一时点，否则恢复后该测试会明确失败。

运行中的数据库文件不能仅靠复制 `.db` 作为一致备份。当前自动迁移也使用 SQLite Backup API，会包含 WAL 中已提交的数据并进行完整性检查；但它只在检测到架构变更时生成，也不包含模型测试图片等文件。升级前仍应显式执行备份脚本并按上述步骤保存成套文件。

回滚需要匹配的代码版本和数据库结构。新版本已迁移数据库时，直接切换旧代码不一定兼容。先停止目标实例，保留当前状态，再按兼容版本与一致备份恢复，并核对文件路径；不要在另一实例运行时覆盖其数据库，也不要把两套实例的数据库互换。

平台 1.3.0 将本机和远程活动 Run 的资源预约保存在数据库中。正常重启 API 时会先回收已进入终态的遗留预约；本机任务还会按 PID、命令签名和启动时间核验进程，并重建、保留或释放对应预约。中心无法在启动时直接核验远程进程，远程活动预约会原样保留，直到 Agent 上报终态或操作员确认远程进程已结束后解除占用。不要通过删除预约表、替换数据库或手工改 Run 状态来“释放显卡”。同卡互斥和多卡整组预约只覆盖平台受管任务，平台外进程仍需单独检查。CPU 和 MPS 不占 CUDA GPU 预约；本机 CPU Run 可彼此并行，本机 MPS Run 彼此互斥但可与 CUDA Run 并行，远程 CPU/MPS Run 使用节点级独占预约。

官方 Agent 从 2.0.0 开始上报 `training.gpu_parallelism=true`；中心实际以该字段为并行开关，并要求设备清单具有稳定 UUID，再向任务下发有效 `gpu_assignment`。未上报字段的官方旧 Agent 继续按整机串行，这是兼容保护，不是调度故障；版本号本身不会代替能力上报，仅升级 Agent 也不会使旧中心获得 GPU 预约协议。升级时保留各节点自己的配置、Token、路径和 `.training-state/`，替换脚本后执行 `--version`、`--check-config` 和 `--once`，再重启对应 Agent 服务。

## 健康检查与故障定位

```bash
sudo systemctl status yolo-platform-dev
sudo journalctl -u yolo-platform-dev -n 100 --no-pager
curl --fail http://127.0.0.1:18088/api/health
```

使用版将服务名替换为 `yolo-platform-prod`，端口替换为 `18089`。健康接口包含 `status`、`instance`、`version`、`scheduler`、`scheduler_error`、`scheduler_last_error`、`scheduler_last_error_at`、`database`、数据库查询耗时和 `runs_root`。数据库失败返回 503；调度循环当前一轮失败时记录异常并返回 `scheduler=error`，后续一轮成功后回到 `running`，最近错误和时间仍保留供排查。这能识别持续报错的调度循环，但仍需结合队列状态和实际进程判断单个训练的进展。耗时至少 1 秒的 `/api/` 请求会输出 `slow_request`，本机删除线程也记录清理情况。

| 现象 | 核查重点 |
| --- | --- |
| 服务启动失败 | `.env` 是否可读、项目/Python/YOLO 路径、运行账户权限、端口占用 |
| 健康正常但任务长期排队 | 检查同一执行节点上的活动任务、GPU 预约和调度日志；远程还要核对 Agent 版本、`last_seen_at`、`training.ready`。使用官方 Agent 时异卡并行应为 2.0.0，并上报 `training.gpu_parallelism=true`、提供稳定 GPU UUID；远程同编号 GPU 不会阻塞本机任务 |
| 本机重启后任务中断 | 平台会按完整命令和进程启动时间尝试重新接管原进程；匹配失败才标为已中断，应先核对事件、PID、日志和检查点再恢复 |
| 远程长期“启动中” | 先确认 Agent 达到当前远程训练最低版本 1.8.0，并检查 `training.ready`、预约和确认重试日志；排查异卡并行时再确认官方 Agent 2.0.0、`training.gpu_parallelism=true`、GPU UUID、中心分配、`.training-state/active/` 和主机锁。未确认领取在 90 秒后可重新分发 |
| 远程长期“正在停止” | 先在远程服务器确认当前 Attempt 的 YOLO 进程已结束；状态保持 `stopping` 至少 30 秒后，可在训练画布使用“解除占用”。该操作只改中心状态，不会终止远程进程 |
| 清理记录迟迟未消失 | 永久删除为异步 202；检查回收站原因、Agent 接入和对应后台操作。输出根目录缺失、允许目录失效或挂载离线时会明确失败并保留记录，恢复路径后重试 |
| 恢复后模型测试有记录但图片打不开 | 数据库与 `model-test-history/` 不属于同一备份时点，或历史目录未恢复；使用成套备份恢复，并核对 `PLATFORM_DATA_ROOT` 和目录权限 |
| 远程指定模型一直准备失败 | 确认 Agent 为 2.0.3、在线并启用训练操作；在远程服务器执行 `--check-config`，确认 `.pt` 位于 `model_test_roots` 内且路径没有符号链接。大模型只在最终校验超时时，调整 `model_transfer_finalize_timeout_seconds` |
| 远程指定模型返回 507 | 检查 `PLATFORM_REMOTE_MODEL_CACHE_QUOTA_BYTES`、`PLATFORM_REMOTE_MODEL_CACHE_MIN_FREE_BYTES` 和 `PLATFORM_DATA_ROOT` 所在磁盘；平台只会自动清理无引用旧版本，不会删除当前或历史引用版本 |
| 日志复制失败或下载不完整 | HTTP 内网剪贴板限制与尾部导出是两个问题；HTTPS 不会把尾部变成完整日志，见 [日志说明](../README.md#日志查看与导出) |

训练命令和输出在任务日志中；API 请求和服务启动信息在 journal 中。更完整的代码证据及改进优先级见 [项目分析](../../docs/project-analysis.md#当前问题与优化优先级)。

## 安全边界

- `.env`、数据库及备份建议仅允许运行账户读取；服务的 `UMask=0077` 会保护新建文件，但不会自动修改旧文件权限。
- 平台管理页面目前没有内置用户登录。只应通过可信内网/VPN访问；跨公网必须在反向代理增加 HTTPS 和身份认证。
- Agent Token 只保护对应节点的 Agent 接口，不等同于网页管理员认证。Token、SSH 密码和真实 `agent-config.json` 不应提交仓库。
- 本机 `PLATFORM_MODEL_TEST_ROOTS` 与远程 `model_test_roots` 只能放可信 `.pt`；PyTorch / Ultralytics 加载权重时可能执行其中携带的代码，路径和摘要校验不能安全地加载恶意权重。
- 节点有等待或进行中的模型传输时不能禁用或移除，进行中的传输也会阻止 Token 轮换。传输进入终态后可以移除节点；中心的模型缓存和历史不会随节点删除，已固定的 `ready` 版本可继续测试，但要抓取新版本必须重新接入并登记新的节点来源。
- 两套平台连接同一训练服务器时，为每套平台创建独立节点、Token 和 Agent 配置，并统一使用当前 Agent 2.0.7 及同一专用系统账户。新中心下发有效 `gpu_assignment` 后，同一账户下的 Agent 才会通过权限为 `0700` 的用户私有状态目录共享 GPU UUID 锁，并以共享方式持有 workspace 与层级输出根锁；同卡互斥由 UUID 锁保证，不同 GPU Run 可以共享配置输出根。旧中心或没有 GPU 分配的任务仍走传统独占锁路径，混合部署必须保留能够覆盖这些旧协议任务的 `training_lock_file` 或等效互斥范围。不同账户无法共享这些宿主机锁和永久删除 owner 保护。训练子进程继承锁，Agent 意外退出后也会保持到训练结束；删除操作独占重叠输出范围，共享输出根存在其他存活 owner 时也会拒绝永久删除。这些锁不限制中心本机调度器或平台外进程。两个 Agent 还应各自核对 `model_test_roots` 和 `model_transfer_finalize_timeout_seconds`，不要为了方便而扩大到整个项目或共享文件系统根。

### 1.4.10 升级说明

本次调整平台前端的指标曲线、五类任务指标详情和续训提示，并将本机 CSV 的非有限指标值统一规范为空值。更新源码后必须重新执行 `npm run build`，然后重启平台后端；否则页面或健康接口仍可能显示旧版本。本机和远程任务使用同一套页面逻辑。Agent 2.0.7 使远程 CSV 对非有限 Epoch 和异常耗时使用与本机一致的容错语义；通信协议没有变化。替换远程 Agent 脚本时保留原 `agent-config.json`、Token 和 `.training-state/`。

### 1.4.9 升级说明

更新平台前后端，重新构建前端并由维护者安排服务重启。完整远程日志下载与分类训练适配需要同步部署 Agent 2.0.6；只替换分发脚本和说明，保留各节点 `agent-config.json`、Token、实际目录和 `.training-state/`。现有训练和远程清理协议保持兼容。

单份日志大小、日志缓存配额和磁盘余量分别由 `PLATFORM_LOG_DOWNLOAD_MAX_BYTES`、`PLATFORM_REMOTE_LOG_CACHE_QUOTA_BYTES`、`PLATFORM_REMOTE_LOG_MIN_FREE_BYTES` 配置，默认 256 MiB / 2 GiB / 2 GiB；均为非负字节数，`0` 关闭对应限制。


### 1.4.11 生命周期升级

本地训练停止宽限期由 `TRAINING_STOP_TIMEOUT_SECONDS` 配置，默认 15 秒，范围限制为 1～300 秒；远程使用 Agent 配置 `training_stop_timeout_seconds`。接管进程在发送信号前仍核验身份。

升级后端前备份数据库和相关文件。启动迁移增加可空 `queue_reason` 与默认空对象 `checkpoint_source`，迁移自动生成数据库备份，旧 Run 保持原有状态。重启开发版后端后，新的预检查接口和排队原因才会生效。Agent 2.0.8 在远程服务器训练空闲时单独升级；只更新两个分发包不代表远程运行进程已升级。
