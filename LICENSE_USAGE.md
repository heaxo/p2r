# 授权使用说明

## 授权文件位置

默认授权文件名为 `license.key`。

- 开发运行：放在项目根目录。
- 打包后运行：放在程序 exe 所在目录。
- 替换 `license.key` 后，程序会在下一次后台校验时自动重新读取；重启程序也会立即生效。

## 生成授权文件

在你自己的电脑或发版环境运行，不要把 `scripts/generate_license.py` 发给客户。

```powershell
python scripts/generate_license.py --days 30 --customer "客户名称" --out license.key
```

指定开始时间：

```powershell
python scripts/generate_license.py --days 30 --start 2026-05-29T00:00:00Z --out license.key
```

## 解码校验授权文件

```powershell
python scripts/decode_license.py license.key
```

输出内容会包含授权天数、开始时间、过期时间和剩余天数。

## 时间回拨处理

程序会在本机记录签名状态，保存“见过的最大时间”。如果客户把系统时间调回到之前的时间，后端会拒绝请求，桌面端显示“无法使用”。

纯离线授权无法在用户删除全部本地状态文件的情况下做到绝对防回拨；要做到强防护，需要联网校验或硬件可信时间源。
