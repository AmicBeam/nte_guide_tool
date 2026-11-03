# 异能对决安卓安装包

现有网站的轻量 WebView 外壳，默认入口为 `https://amicbeam.fun/card-game`。网站代码、规则、账号、联机、人机和存档继续由 Flask 服务提供。安装包需要联网，浏览器与应用的登录态分别保存。网站内容更新通常无需重新安装；外壳或入口域名变更需要重新打包。

## 构建

需要 Python 3、JDK 17 或更新版本、Android SDK（Platform 36、Build Tools 36.1.0）及网络。首次构建下载固定版本的 Gradle 与构建插件；缺少平台时，已接受许可的 SDK 可自动安装。Android Studio 可打开本目录。命令行设置 `JAVA_HOME` 和 `ANDROID_HOME`；macOS 脚本会识别默认 Android Studio JDK 和 SDK 路径。

在仓库根目录执行，默认打开线上大厅：

```bash
python3 android/build_apk.py
```

产物 `dist/android/esper-duel-release.apk` 可直接安装分享。脚本创建并复用 `~/.local/share/nte-esper-duel-android/` 内的签名密钥与密码文件。必须安全备份整个目录，覆盖安装需要同一签名；不要提交或分享这些文件。可设置 `NTE_ANDROID_SIGNING_PROPERTIES` 使用仓库外的现有签名配置，字段为 `storeFile`、`storePassword`、`keyAlias`、`keyPassword`。

调试包使用不同应用 ID，可与正式版并存：

```bash
python3 android/build_apk.py --debug --url http://10.0.2.2:5001/card-game
```

`10.0.2.2` 是模拟器访问宿主机的地址，真机联调用电脑局域网地址。产物为 `dist/android/esper-duel-debug.apk`。直接执行 `./gradlew :app:assembleDebug` 默认使用线上地址。正式包拒绝 HTTP 入口且禁用明文流量，不提供服务器切换界面。

## 行为与边界

- 复用异能对决图标；横屏、沉浸显示、前台屏幕常亮，避让系统遮挡与键盘。
- 开启 JavaScript、DOM Storage 和第一方 Cookie，复用登录、构筑和页面设置；不提供 JavaScript 原生桥，不读取浏览器登录资料。
- 关闭再打开保留有效登录凭证，期限沿用网站配置（默认 72 小时）；不在外壳保存账号原始密码，到期后重新登录。
- 同源导航留在应用内，外站 HTTP(S) 链接由系统浏览器打开，其他协议不由外壳处理。
- 页面加载失败提供重新连接和返回大厅。局内接口错误和重连沿用网页处理，不自动重发出牌请求。
- 返回键先退出网页全屏，再按网页历史返回，没有历史时确认退出。切回前台不强制刷新；系统回收后以网站已保存状态恢复。
- 复用 WebView 缓存，不打包全站素材；不提供离线对局、后台保活、推送和原生文件下载／分享。网页文本复制与文件导入导出需分别验证兼容性。
- 普通手机使用横屏，大屏或较新系统可能按系统窗口策略调整方向与尺寸。

## 验证

```bash
cd android
./gradlew :app:assembleDebug :app:lintDebug
```

安装后检查：首次跳转登录、登录回到大厅、重启保留有效登录、构筑／图鉴与牌桌触控、返回键、后台恢复、启动断网后的重试。调试包允许 WebView 调试，正式包关闭调试并禁用系统备份。正式包签名可用 SDK 的 `apksigner verify --verbose` 检查。
