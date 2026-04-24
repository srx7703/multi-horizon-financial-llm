# TPU 服务器 SSH 连接指南（协作版）

本文档帮助协作者从零开始连接我们共享的 TPU 服务器。

---

## 服务器信息

| 项目 | 值 |
|------|------|
| **GCP 项目** | `tpu-tabnet` |
| **节点名称** | `tpu-v5e-64-vm-eu-b` |
| **区域 (Zone)** | `europe-west4-b` |
| **加速器类型** | `v5litepod-64` (TPU v5e, 64 核) |
| **网络模式** | 仅内网 IP（需通过 IAP 隧道连接） |

---

## 第一步：安装 Google Cloud SDK

### macOS
```bash
# 使用 Homebrew 安装
brew install --cask google-cloud-sdk
```

### Linux (Ubuntu/Debian)
```bash
# 添加 Cloud SDK 源
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" | \
    sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo tee /usr/share/keyrings/cloud.google.asc

sudo apt-get update && sudo apt-get install google-cloud-cli
```

### Windows
从 [Google Cloud SDK 官网](https://cloud.google.com/sdk/docs/install) 下载安装包。

### 验证安装
```bash
gcloud --version
```

---

## 第二步：初始化 gcloud 并登录

```bash
# 1. 初始化（会打开浏览器让你登录 Google 账号）
gcloud init

# 2. 登录后设置默认项目
gcloud config set project tpu-tabnet

# 3. 安装 alpha 组件（TPU 命令需要）
gcloud components install alpha
```

---

## 第三步：获取项目权限

> **此步骤需要项目 Owner（杰力）操作，请将你的 Google 账号邮箱发给我。**

### Owner 操作步骤：

1. 打开 [GCP IAM 控制台](https://console.cloud.google.com/iam-admin/iam?project=tpu-tabnet)
2. 点击顶部 **「授予访问权限」** 按钮
3. 在 **「新的主账号」** 中输入协作者的 Google 邮箱
4. 添加以下角色（逐个添加）：

| 角色 | 用途 |
|------|------|
| `Compute OS Login` | 允许通过 OS Login 方式 SSH 登录 |
| `IAP-secured Tunnel User` | 允许通过 IAP 隧道连接内网服务器 |
| `TPU Viewer` | 查看 TPU 资源状态 |
| `Service Account User` | 使用服务账号（SSH 需要） |

5. 点击 **「保存」**

### 验证权限
协作者可以运行以下命令验证是否有权限：
```bash
gcloud alpha compute tpus queued-resources list --project=tpu-tabnet --zone=europe-west4-b
```
如果能看到资源列表，说明权限已生效。

---

## 第四步：SSH 连接服务器

```bash
gcloud alpha compute tpus tpu-vm ssh tpu-v5e-64-vm-eu-b \
    --zone=europe-west4-b \
    --project=tpu-tabnet \
    --tunnel-through-iap
```

### 参数说明
- `tpu-v5e-64-vm-eu-b`：TPU VM 的节点名称
- `--zone=europe-west4-b`：服务器所在区域（欧洲西部）
- `--project=tpu-tabnet`：GCP 项目 ID
- `--tunnel-through-iap`：**必须加此参数**，因为服务器只有内网 IP，需要通过 Google IAP (Identity-Aware Proxy) 隧道才能连接

### 首次连接
首次连接时会自动生成 SSH 密钥并传播到 TPU worker，可能需要等待 1-2 分钟，看到类似以下输出即为正常：
```
Propagating SSH public key to all TPU workers...
...........................................................................done.
SSH: Attempting to connect to worker 0...
```

---

## 快速测试连接（不进入交互式 shell）

```bash
gcloud alpha compute tpus tpu-vm ssh tpu-v5e-64-vm-eu-b \
    --zone=europe-west4-b \
    --project=tpu-tabnet \
    --tunnel-through-iap \
    --command="hostname && echo 'Connection OK'"
```

---

## 常见问题排查

### 1. SSH 连接超时
```
ERROR: (gcloud.alpha.compute.tpus.tpu-vm.ssh) Could not SSH into the instance.
```
**原因**：没有加 `--tunnel-through-iap` 参数，或 IAP 权限未配置。
**解决**：确保命令中包含 `--tunnel-through-iap`，并检查是否有 `IAP-secured Tunnel User` 角色。

### 2. 权限不足 (403 Forbidden)
```
ERROR: (gcloud.alpha.compute.tpus.tpu-vm.ssh) PERMISSION_DENIED
```
**原因**：Google 账号未被添加到项目中，或缺少必要角色。
**解决**：联系项目 Owner 按第三步添加权限。

### 3. 找不到资源 (404 Not Found)
```
ERROR: (gcloud.alpha.compute.tpus.tpu-vm.ssh) NOT_FOUND
```
**原因**：节点名称或区域写错了，或服务器已被删除/重建。
**解决**：先运行以下命令确认当前活跃的节点：
```bash
gcloud alpha compute tpus queued-resources list --project=tpu-tabnet --zone=europe-west4-b
```

### 4. gcloud 版本过低
```
ERROR: (gcloud.alpha.compute.tpus) Invalid choice: 'tpu-vm'
```
**原因**：gcloud SDK 版本太旧，不支持 `tpu-vm` 子命令。
**解决**：
```bash
gcloud components update
gcloud components install alpha
```

---

## 联系方式

如有问题，请联系项目 Owner 获取帮助。
