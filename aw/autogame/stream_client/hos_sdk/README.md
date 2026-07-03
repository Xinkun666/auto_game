*斜体部分需项目Owner自行修改，README将与代码仓保持同步*

[编辑README](https://open.codehub.huawei.com/innersource/oh_remote_device_G/oh_remote_device/files?ref=master&filePath=README.md&isFile=true)

## 项目介绍

*此项目是为了对标安卓设备的scrcpy投屏工具，为开发者提供快捷获取鸿蒙设备视频流的方式，在此基础上开发更为完善的远程真机服务。*

## 项目优势

*画面清晰流畅。*

## 快速入门

### 1.安装
1.1 打包方式
不混淆打包：build.bat d:\python3.11.9\
混淆打包：build.bat d:\python3.11.9\ o
需要指定python.exe路径

1.2 安装打包
python -m pip install  .\dist\hoscrcpy_sdk-1.0.0.0.tar.gz

### 2.使用

#### sdk提供给开发者使用的类的路径在hoscrcpy_sdk下，有以下4个类

#### 1.HosRemoteDevice

| 方法                                                         | 参数                             | 解释                                                         |
| ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| HosRemoteDevice                                      | config                               | 构造函数，通过创建HosRemoteConfig对象来初始化HosRemoteDevice对象，可以设置获取视频流分辨率的缩放倍率、帧率、码率等配置                         |
| start_capture_screen(self, screen_cap_callback: ScreenCapCallback)      | screen_cap_callback                | 通过传入视频流回调函数来开始获取视频流                       |
| stop_capture_screen(self)                                              |                                  | 停止获取视频流                                               |
| get_sn(self)                                                      |                                  | 获取当前ScrcpyDevice的sn                                     |
| execute_shell_command(self, command: Union[str, list], timeout: int = 5 * 60)             | command， timeout                | 让设备执行hdc shell命令。command：要执行的shell命令，timeout：shell命令超时时间，单位秒 |
| get_screen_size(self, need_update: bool)                            | need_update                       | 获取当前设备的分辨率，传入true时会重新获取分辨率信息，传入false时会使用之前的缓存分辨率信息 |
| on_touch_down(self, x: int, y: int)         | x,y    | 注入手指按下事件, xy为手指按下的坐标(需要调用startCaptureScreen方法后再调用) |
| on_touch_up(self, x: int, y: int)           | x,y    | 注入手指抬起事件, xy为手指抬起的坐标(需要调用startCaptureScreen方法后再调用) |
| on_touch_move(self, x: int, y: int)         | x,y    | 注入手指移动事件, xy为手指移动的坐标(需要调用startCaptureScreen方法后再调用) |
| set_rotation_horizontal(self)           |        | 设置设备屏幕为横屏状态(需要调用startCaptureScreen方法后再调用)           |
| set_rotation_vertical(self)             |        | 设置设备屏幕为竖屏状态(需要调用startCaptureScreen方法后再调用)           |

#### 2.ScreenCapCallback

| 方法                          | 参数       | 解释                                                         |
| ----------------------------- | ---------- | ------------------------------------------------------------ |
| on_data(self, byte_buffer: bytes) | byte_buffer | 当sdk获取到视频流数据后，会回调此方法，并传入视频流的ByteBuffer，开发者可以根据此ByteBuffer进行画面显示 |
| on_exception(self, err: Exception)    | err        | 当获取视频流出错后，会回调此方法，传入报错信息               |
| on_ready()                     |            | 因为需要设备画面变动才能获取视频流，所以如果设备已经处于一个亮屏且画面没有变动的状态时，onData方法是不会被调用的。on_ready方法是为了通知开发者当前已经处于了获取视频流就绪状态，开发者可以在此做一些使设备画面变动的动作，比如：按电源键点亮关闭屏幕 |

#### 3.HosRemoteConfig

| 方法                                               | 参数              | 解释                                                                                                                               |
|--------------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------|
| HosRemoteConfig(sn)                              | sn              | sn  手机序列号                                                                                                                        |
| set_scale(self, scale: int)                      | scale           | 视频流分辨率的缩放倍率(输入2代表获取的视频流分辨率为原来的二分之一，3代表为原来的三分之一；最大设置为5)                                                                           |
| set_frame_rate(self, frame_rate: int)            | frame_rate      | 因为需要设备画面变动才能获取视频流，所以如果设备已经处于一个亮屏且画面没有变动的状态时，onData方法是不会被调用的。onReady方法是为了通知开发者当前已经处于了获取视频流就绪状态，开发者可以在此做一些使设备画面变动的动作，比如：按电源键点亮关闭屏幕 |
| set_bit_rate(self, bit_rate: int)                | bit_rate        | 视频流码率,单位Mbps,默认30Mbps                                                                                                            |
| set_port(self, port: int)                        | port            | 用以设置设备侧的转发端口,默认5000,无特殊情况不需要配置                                                                                                   |
| set_use_old_version(self, use_old_version: bool) | use_old_version | use_old_version表示是否使用旧的投屏so，默认为false                                                                                             |

#### 4.Size

| 属性   | 解释             |
| ------ | ---------------- |
| width  | 设备分辨率的宽度 |
| height | 设备分辨率的长度 |

### 3.样例
此sdk主要是为前端网页查看设备所使用，前端有个名为：jmuxer 的库，能够直接解析h264视频流的ByteBuffer并进行画面的显示

可以通过创建WebSocket给前端传输视频流数据，以下为前端和后端的示例

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description"
          content="jMuxer - a simple javascript mp4 muxer for non-standard streaming communications protocol">
    <meta name="keywords" content="h264 player, mp4 player, mse, mp4 muxing, jmuxer, aac player">
    <title>JMuxer demo</title>
</head>

<body>
<div id="" class="container">
    <video class="left" autoplay style="cursor: pointer; width: 500px; height: 800px; text-align: center;" id="player"></video>
</div>
</body>

<script type="text/javascript" src="jmuxer.min.js"></script>
<script>
    var jmuxer = new JMuxer({
        node: 'player',
        mode: 'video',
        flushingTime: 0,
        fps: 60,
        debug: false,
        onError: function (data) {
            jmuxer.reset();
        }
    });
    var videoPlayer = document.getElementById("player");
    var socketURL = 'ws://127.0.0.1:8888/自己设备的sn';
    var flag = 0;
    var ws = new WebSocket(socketURL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = function () {
        var txt = '{"type":"screen","message":""}';
        ws.send(txt)
    }
    //oh设备video监听
    videoPlayer.addEventListener("mousedown", doMouseDown, false);
    videoPlayer.addEventListener('mouseup', doMouseUp, false);
    videoPlayer.addEventListener('mousemove', doMouseDrag, false);


    ws.onmessage = function (message) {
        if (typeof message.data === "string") {
            console.log(message.data)
            var obj = JSON.parse(message.data);
            if (obj.type === "recordEvent") {
                addRecordMsg(obj.message)
            }
        } else if (message.data instanceof ArrayBuffer) {
            jmuxer.feed({
                video: new Uint8Array(message.data)
            });
        }
    }

    ws.onclose = function (event) {
        console.log("close")
    }

    ws.onerror = function (event) {
        console.log("error")
    }


    function getActualClickPoint(x, y) {
        // 获取video控件的长宽
        var videoHeight = videoPlayer.offsetHeight
        var videoWidth = videoPlayer.offsetWidth
        // 设备分辨率(一般视频流分辨率都与设备原始分辨率相同)
        var streamHeight = videoPlayer.videoHeight
        var streamWidth = videoPlayer.videoWidth
        // 计算视频流宽高比
        var aspectRatio = streamWidth / streamHeight
        //  计算等比例缩放后的视频区域大小
        var streamSizeWidth = Math.min(videoWidth, aspectRatio * videoHeight)
        var streamSizeHeight = streamSizeWidth / aspectRatio
        // 获取视频流在video控件内的左上角起始点
        var startX = 0.5 * (videoWidth - streamSizeWidth)
        var startY = 0.5 * (videoHeight - streamSizeHeight)
        // 判断点击点是否在范围内,超出边界的返回最边界的坐标
        if (x < startX) {
            x = startX + 2;
            y = startY + (streamSizeHeight / 2);
        } else if (x > startX + streamSizeWidth) {
            x = startX + streamSizeWidth - 2;
            y = startY + (streamSizeHeight / 2);
        } else if (y < startY) {
            x = startX + (streamSizeWidth / 2);
            y = startY + 2;
        } else if (y > startY + streamSizeHeight) {
            x = startX + (streamSizeWidth / 2);
            y = startY + streamSizeHeight - 2;
        }
        // 计算video中显示的视频大小相对于原始视频流大小的缩放比
        var widthScaleRate = streamSizeWidth / streamWidth
        var heightScaleRate = streamSizeHeight / streamHeight
        var actualX = Math.floor((x - startX) / widthScaleRate)
        var actualY = Math.floor((y - startY) / heightScaleRate)
        return {
            actualX: actualX,
            actualY: actualY
        };
    }


    function doMouseDrag(event) {
        if (flag !== 1) {
            return;
        }
        var x = event.offsetX;
        var y = event.offsetY;
        var clickPoint = getActualClickPoint(x, y)
        if (clickPoint === null) {
            return;
        }
        // 发送点击位置
        var message = '{"event":"move","x":' + clickPoint.actualX + ', "y":' + clickPoint.actualY + '}';
        var txt = '{"type":"touchEvent","message":' + message + '}';
        ws.send(txt)
    }


    function doMouseDown(event) {
        lastMouseDownTime = new Date().getTime();
        lastMouseDownX = event.offsetX;
        lastMouseDownY = event.offsetY;
        var clickPoint = getActualClickPoint(lastMouseDownX, lastMouseDownY)
        if (clickPoint === null) {
            return;
        }
        flag = 1;
        // 发送点击位置
        var message = '{"event":"down","x":' + clickPoint.actualX + ', "y":' + clickPoint.actualY + '}';
        var txt = '{"type":"touchEvent","message":' + message + '}';
        ws.send(txt);
    }

    function doMouseUp(event) {
        flag = 0;
        var x = event.offsetX;
        var y = event.offsetY;
        var clickPoint = getActualClickPoint(x, y)
        if (clickPoint === null) {
            return;
        }
        // 发送点击位置
        var message = '{"event":"up","x":' + clickPoint.actualX + ', "y":' + clickPoint.actualY + '}';
        var txt = '{"type":"touchEvent","message":' + message + '}';
        ws.send(txt)
    }
</script>
</html>
```

```python
import asyncio
import websockets
from ScreenCapCallback import ScreenCapCallback
from HosRemoteDevice import HosRemoteDevice
from HosRemoteConfig import HosRemoteConfig
import json


async def echo(websocket, path):
    async for message in websocket:
        # print(f"Received: {message}")
        json_message = json.loads(message)
        if json_message["type"] == "screen":
            class VideoStreamTest(ScreenCapCallback):

                def __init__(self):
                    super(VideoStreamTest, self).__init__()

                def on_data(self, byteBuffer):

                    async def send_data(frame_data):
                        await websocket.send(frame_data)

                    asyncio.run(send_data(byteBuffer))

                def on_exception(self, err):
                    print("重写onException")

                def on_ready(self):
                    print("重写onReady")

            screen_cap_callback = VideoStreamTest()
            host_device.start_capture_screen(screen_cap_callback)

        elif json_message["type"] == "touchEvent":
            x = json_message["message"]["x"]
            y = json_message["message"]["y"]
            event = json_message["message"]["event"]
            if event == "down":
                host_device.on_touch_down(x, y)
            elif event == "up":
                host_device.on_touch_up(x, y)
            elif event == "move":
                host_device.on_touch_move(x, y)

        elif json_message["type"] == "keyEvent":
            pass

# send_video_data()
if __name__ == '__main__':
    config = HosRemoteConfig(sn='****')
    host_device = HosRemoteDevice(config)
    start_server = websockets.serve(echo, "localhost", 8888)
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt as e:
        host_device.stop_capture_screen()

```

## 如何反馈

*联系人：谭广 60079072，利广杰 00418355。*

## 如何贡献代码

###  Git MM工具安装

由于该代码仓托管在通用区，需要使用git MR来提交代码。

黄区windows的工具安装地址为：

- TortoiseGit : http://3ms.huawei.com/hi/group/2031557/file_15351180.html
- git : http://pages.huawei.com/codeclub/guides/git_install_windows
- git-mm ： Git Bash 中执行 curl -k https://isource-pages.huawei.com/iSource/git-mm/current/win64/git-mm.exe -o /usr/bin/git-mm.exe

绿区可参考 https://his.huawei.com/doc/#/page.html?service_code=hrn:his:servicemarket::service:codehub&group_id=496c3432d79a4d7292cb9b7caa8a2489&lang=zh_CN

注意： 安装git-mm后， 执行一个 git mm，看看是否安装成功。

### 使用Git MR 提交代码

Git MR提交代码的工作流， 如下图

![image.png](https://file.openx.huawei.com/openx-file/dfsfile/04/96/Chx5KmEp5R-EIbXiAAAAACCRD5A016.png)



* git Commit命令本地提交

  ![image.png](https://file.openx.huawei.com/openx-file/dfsfile/04/96/Chx5KmEp6jaESyAFAAAAALvXyUQ851.png)

* 用 git mr命令完成推送代码到项目仓

  ![image.png](https://file.openx.huawei.com/openx-file/dfsfile/04/96/Chx5KmEp5wCEY5JcAAAAAOKVHac347.png)

* 更多参考 https://openx.huawei.com/communityHome/postDetail?postId=3180&id=32
