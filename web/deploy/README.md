# 双实例部署与维护

本机有两套彼此独立的平台。开发版固定使用 `18088`，正式使用版固定使用 `18089`。两套服务必须使用不同的 systemd 服务名，不能都安装成 `yolo-platform.service`。

仓库同时保存 `yolo-platform-dev.service` 和 `yolo-platform-prod.service`，两套 checkout 中内容一致。环境差异放在不同文件中，不再直接修改同一个 service 文件，因此后续 `git pull` 不会因开发/正式路径不同而再次冲突。

| 实例 | 项目目录 | 端口 | systemd 建议名称 | 默认数据库 |
| --- | --- | --- | --- | --- |
| 开发版 | `/data/yolo-模型训练平台开发/yolo` | `18088` | `yolo-platform-dev.service` | 本项目 `web/data/platform.db` |
| 使用版 | `/data/yolo-模型训练平台开发/yolo使用版/yolo` | `18089` | `yolo-platform-prod.service` | 本项目 `web/data/platform.db` |

`YOLO_PROJECT_ROOT=.` 与相对的 `DATABASE_URL=sqlite:///web/data/platform.db` 都由后端按代码所在项目解析，不受启动终端目录影响。不要让两套 `.env` 指向同一个绝对数据库、数据目录或运行目录。

## 安装服务文件

以下命令只复制和校验服务；最后的启动由管理员自行执行。更新前先确认没有本机或远程训练正在运行。

```bash
sudo cp /data/yolo-模型训练平台开发/yolo/web/deploy/yolo-platform-dev.service \
  /etc/systemd/system/yolo-platform-dev.service
sudo cp /data/yolo-模型训练平台开发/yolo使用版/yolo/web/deploy/yolo-platform-prod.service \
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

健康接口中的 `instance` 应分别为 `development` 和 `production`。服务使用单个 Uvicorn worker；调度器、SQLite 和后台删除任务都依赖单实例写入，不能通过增加 worker 数量扩容。

## 更新与备份顺序

1. 确认没有训练运行，也没有显示“清理中”的任务。
2. 在对应项目执行数据库在线备份：

   ```bash
   /data/conda/envs/yolo/bin/python web/deploy/backup_database.py
   ```

3. 拉取或复制代码，构建前端并运行测试。
4. 只重启需要更新的那个服务，不要同时覆盖两套 `.env` 或 `web/data/`。
5. 检查健康接口、服务日志和首页，再更新另一实例。

备份脚本使用 SQLite Backup API，可正确包含 WAL 中已提交的数据，输出到当前项目的 `web/data/backups/`，权限为 600。它不会自动删除旧备份。

## 安全边界

- `.env`、数据库及备份建议仅允许运行账户读取；服务的 `UMask=0077` 会保护新建文件，但不会自动修改旧文件权限。
- 平台管理页面目前没有内置用户登录。只应通过可信内网/VPN访问；跨公网必须在反向代理增加 HTTPS 和身份认证。
- Agent Token 只保护对应节点的 Agent 接口，不等同于网页管理员认证。Token、SSH 密码和真实 `agent-config.json` 不应提交仓库。
- 生产实例和开发实例连接同一训练服务器时，必须为每套平台创建独立节点、独立 Token 和独立 Agent 目录，并升级到 Agent 1.4.0。两个 Agent 的 `training_cwd` 相同时会共享主机训练锁，避免同时占用 GPU；任务记录和产物仍分别归属各自平台。
