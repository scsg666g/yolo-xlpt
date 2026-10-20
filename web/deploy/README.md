# 双实例部署与维护

本仓库的服务模板按两套独立 checkout 配置：开发版使用 `18088`，使用版使用 `18089`。两套服务必须使用不同的 systemd 服务名，不能都安装成 `yolo-platform.service`。模板存在不代表两套服务已在目标机器安装或运行。

安装依赖和构建步骤见 [项目首页](../../README.md)；操作规则见 [Web 说明](../README.md)；远程服务器见 [Agent 说明](../agent/README.md)；架构与故障证据见 [项目分析](../../docs/project-analysis.md)。当前开发平台版本为 **1.3.0**，配套远程 Agent 为 **2.0.0**。

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
| `PLATFORM_MODEL_TEST_HISTORY_QUOTA_BYTES`、`PLATFORM_MODEL_TEST_TASK_QUOTA_BYTES` | 默认 20 GiB / 5 GiB | 中心保存的模型测试历史总量和单任务量；单位为字节，设为 `0` 才关闭对应配额 |
| `PLATFORM_MODEL_TEST_MIN_FREE_BYTES`、`PLATFORM_MODEL_TEST_IMAGE_RESERVE_BYTES` | 默认 2 GiB / 64 MiB | 创建测试时保留的最低可用空间和每张图片的保守预留；单位为字节 |
| `DATABASE_URL` | `sqlite:///web/data/platform.db` | 独立于数据根目录，修改 `PLATFORM_DATA_ROOT` 不会自动修改数据库地址 |
| `YOLO_BIN`、`ULTRALYTICS_ROOT` | 训练程序与源码配置目录 | 参数目录来自源码，不是对实际 CLI 能力的探测 |
| `PLATFORM_INSTANCE_NAME` | 模板分别设置 `development`、`production`，但 `.env` 中同名项优先 | 将示例的 `local` 改为对应实例名称，用健康接口核对最终生效值 |
| `PLATFORM_VERSION` | 默认读取 checkout 根目录 `VERSION` | 通常不覆盖；仅打包无 `VERSION` 的运行环境才需要显式设置 |

目录配置变更不会迁移旧文件，也不会自动更新数据库记录里的绝对路径。迁移须同时处理数据库、运行目录、模型、数据集和远程 Agent 的中心地址。

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

脚本不会自动删除旧备份，也不包含模型、数据集、产物、日志或 `model-test-history/`。SQLite 快照中的 `model_tests`、`model_test_images` 只是模型测试索引和元数据；对应原图、结果和下载文件位于 `model-test-history/`。Backup API 不会冻结这些文件的并发变化，因此需要可完整恢复且相互匹配的模型测试历史时，必须在目标服务停止或静默期间依次生成数据库快照和复制历史目录。按上述步骤确认没有活动模型测试后，`model-test-work/` 不包含需要备份的持久结果；若在活动测试期间强行备份，有效的固定权重副本也必须与数据库状态保持同一时点，否则恢复后该测试会明确失败。

运行中的数据库文件不能仅靠复制 `.db` 作为一致备份；当前自动迁移中的备份使用 `shutil.copy2`，在 WAL 模式下不能替代升级前执行的 SQLite Backup API 备份。

回滚需要匹配的代码版本和数据库结构。新版本已迁移数据库时，直接切换旧代码不一定兼容。先停止目标实例，保留当前状态，再按兼容版本与一致备份恢复，并核对文件路径；不要在另一实例运行时覆盖其数据库，也不要把两套实例的数据库互换。

平台 1.3.0 将本机和远程活动 Run 的资源预约保存在数据库中。正常重启 API 时，启动校准会保留仍有有效运行身份的预约，并回收已经终止的遗留预约；不要通过删除预约表、替换数据库或手工改 Run 状态来“释放显卡”。同卡互斥和多卡整组预约只覆盖平台受管任务，平台外进程仍需单独检查。CPU 和 MPS 任务不占 GPU 预约；远程 CPU/MPS Run 使用节点级串行预约，本机 CPU/MPS Run 不申请 GPU 资源。

远程同节点不同 GPU 并行需要 Agent 2.0.0 上报 `training.gpu_parallelism=true`。旧 Agent 或缺少该字段的节点会继续按整机串行，这是兼容保护，不是调度故障。升级 Agent 时保留各节点自己的配置、Token 和 `.training-state/`，替换脚本后执行 `--version`、`--check-config` 和 `--once`，再重启对应 Agent 服务。

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
| 健康正常但任务长期排队 | 检查同一执行节点上的活动任务、GPU 预约和调度日志；远程还要核对 Agent 版本、`last_seen_at`、`training.ready`。异卡并行需要 Agent 2.0.0 上报 `training.gpu_parallelism=true`；远程同编号 GPU 不会阻塞本机任务 |
| 本机重启后任务中断 | 平台会按完整命令和进程启动时间尝试重新接管原进程；匹配失败才标为已中断，应先核对事件、PID、日志和检查点再恢复 |
| 远程长期“启动中” | 确认平台 1.3.0 与 Agent 2.0.0 已配套部署，再检查 `training.ready`、`training.gpu_parallelism`、GPU 预约、Agent 确认重试日志、`.training-state/active/` 和主机锁；未确认领取在 90 秒后可重新分发 |
| 远程长期“正在停止” | 先在远程服务器确认当前 Attempt 的 YOLO 进程已结束；状态保持 `stopping` 至少 30 秒后，可在训练画布使用“解除占用”。该操作只改中心状态，不会终止远程进程 |
| 清理记录迟迟未消失 | 永久删除为异步 202；检查回收站原因、Agent 接入和对应后台操作。输出根目录缺失、允许目录失效或挂载离线时会明确失败并保留记录，恢复路径后重试 |
| 恢复后模型测试有记录但图片打不开 | 数据库与 `model-test-history/` 不属于同一备份时点，或历史目录未恢复；使用成套备份恢复，并核对 `PLATFORM_DATA_ROOT` 和目录权限 |
| 日志复制失败或下载不完整 | HTTP 内网剪贴板限制与尾部导出是两个问题；HTTPS 不会把尾部变成完整日志，见 [日志说明](../README.md#日志查看与导出) |

训练命令和输出在任务日志中；API 请求和服务启动信息在 journal 中。更完整的代码证据及改进优先级见 [项目分析](../../docs/project-analysis.md#当前问题与优化优先级)。

## 安全边界

- `.env`、数据库及备份建议仅允许运行账户读取；服务的 `UMask=0077` 会保护新建文件，但不会自动修改旧文件权限。
- 平台管理页面目前没有内置用户登录。只应通过可信内网/VPN访问；跨公网必须在反向代理增加 HTTPS 和身份认证。
- Agent Token 只保护对应节点的 Agent 接口，不等同于网页管理员认证。Token、SSH 密码和真实 `agent-config.json` 不应提交仓库。
- 两套平台连接同一训练服务器时，为每套平台创建独立节点、Token 和 Agent 配置，并统一使用 Agent 2.0.0 及同一专用系统账户。同一账户下的 Agent 通过权限为 `0700` 的用户私有状态目录共享 GPU UUID 锁、层级输出根锁和 workspace owner 登记，不需要手工配成相同 `training_lock_file`。不同账户无法共享这些宿主机锁和永久删除 owner 保护。训练子进程继承锁，Agent 意外退出后也会保持到训练结束；共享输出根存在其他存活 owner 时永久删除会被拒绝。这些锁不限制中心本机调度器或平台外进程。
