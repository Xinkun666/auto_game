class HosRemoteConfig:
    sn = ""
    scale = 1  # 分辨率缩放比
    ip = "127.0.0.1"
    port = 8710
    frame_rate = 120  # 帧率
    bit_rate = 30  # 码率
    device_port = 5000  # 转发端口号
    screen_id = 0  # 屏幕id
    windows_id = ""  # 窗口id
    app_pid = ""  # 应用进程id
    encoder_type = "0"  # 编码器类型
    use_old_version = False  # use_old_version 表示是否使用旧的投屏so，默认为false

    def __init__(self, **kwargs) -> None:
        self.sn = kwargs.get("sn", "")
        self.ip = kwargs.get("ip", "127.0.0.1")
        self.port = kwargs.get("port", 8710)
        self.scale = kwargs.get("scale", 1)
        self.frame_rate = kwargs.get("frame_rate", 120)
        self.bit_rate = kwargs.get("bit_rate", 30)
        self.device_port = kwargs.get("device_port", 5000)
        self.screen_id = kwargs.get("screen_id", 0)
        self.windows_id = kwargs.get("windows_id", "")
        self.app_pid = kwargs.get("app_pid", "")
        self.encoder_type = kwargs.get("encoder_type", "0")
        self.use_old_version = kwargs.get("use_old_version", False)
        if not self.sn:
            raise ValueError("sn cannot be empty")

    def get_params(self) -> str:
        if self.windows_id and self.app_pid:
            return "-scale {} -frameRate {} -bitRate {} -p {} -screenId {} -windowsId {} \
                -appPid {} -encodeType {}".format(self.scale, self.frame_rate, self.bit_rate,
                                                  self.device_port, self.screen_id, self.windows_id, self.app_pid,
                                                  self.encoder_type)
        else:
            return ("-scale {} -frameRate {} -bitRate {} -p {}\
                     -screenId {}  -encodeType {}").format(self.scale, self.frame_rate, self.bit_rate,
                                                           self.device_port, self.screen_id, self.encoder_type)

    def get_sn(self) -> str:
        return self.sn

    def set_sn(self, sn: str) -> None:
        self.sn = sn

    def get_ip(self) -> str:
        return self.ip

    def set_ip(self, ip: str) -> None:
        self.ip = ip

    def get_scale(self) -> int:
        return self.scale

    def set_scale(self, scale: int) -> None:
        self.scale = scale

    def get_frame_rate(self) -> int:
        return self.frame_rate

    def set_frame_rate(self, frame_rate: int) -> None:
        self.frame_rate = frame_rate

    def get_bit_rate(self) -> int:
        return self.bit_rate

    def set_bit_rate(self, bit_rate: int) -> None:
        self.bit_rate = bit_rate

    def get_port(self) -> int:
        return self.port

    def set_port(self, port: int) -> None:
        self.port = port

    def get_device_port(self) -> int:
        return self.device_port

    def set_device_port(self, device_port: int) -> None:
        self.device_port = device_port

    def get_screen_id(self) -> int:
        return self.screen_id

    def set_screen_id(self, screen_id: int) -> None:
        self.screen_id = screen_id

    def get_windows_id(self):
        return self.windows_id

    def set_windows_id(self, windows_id: str) -> None:
        self.windows_id = windows_id

    def get_app_pid(self):
        return self.app_pid

    def set_app_pid(self, app_pid: str) -> None:
        self.app_pid = app_pid

    def get_encoder_type(self):
        return self.encoder_type

    def set_encoder_type(self, encoder_type: str) -> None:
        self.encoder_type = encoder_type

    def get_use_old_version(self) -> bool:
        return self.use_old_version

    def set_use_old_version(self, use_old_version: bool) -> None:
        self.use_old_version = use_old_version
