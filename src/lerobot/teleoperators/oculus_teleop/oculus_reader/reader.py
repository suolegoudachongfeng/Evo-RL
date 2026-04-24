import os
import threading
from pathlib import Path

import numpy as np
from ppadb.client import Client as AdbClient

from .buttons_parser import parse_buttons
from .fps_counter import FPSCounter


class OculusReader:
    def __init__(
        self,
        ip_address: str | None = None,
        port: int = 5555,
        apk_name: str = "com.rail.oculus.teleop",
        print_fps: bool = False,
        run: bool = True,
    ):
        self.running = False
        self.last_transforms: dict[str, np.ndarray] = {}
        self.last_buttons: dict[str, bool | tuple[float, ...]] = {}
        self._lock = threading.Lock()
        self.tag = "wE9ryARX"

        self.ip_address = ip_address
        self.port = port
        self.apk_name = apk_name
        self.print_fps = print_fps
        self.fps_counter = FPSCounter() if self.print_fps else None

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self) -> None:
        self.running = True
        self.device.shell(
            'am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" '
            "-a android.intent.action.MAIN -c android.intent.category.LAUNCHER"
        )
        self.thread = threading.Thread(target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line))
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_network_device(self, client: AdbClient) -> object:
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system("adb devices")
            client.remote_connect(self.ip_address, self.port)
        device = client.device(f"{self.ip_address}:{self.port}")
        if device is None:
            raise RuntimeError(
                "Oculus Quest network device not found. Run `adb shell ip route` on the Quest-connected host "
                "and verify the configured IP address."
            )
        return device

    def get_usb_device(self, client: AdbClient) -> object:
        try:
            devices = client.devices()
        except RuntimeError:
            os.system("adb devices")
            devices = client.devices()
        for device in devices:
            if device.serial.count(".") < 3:
                return device
        raise RuntimeError(
            "Oculus Quest USB device not found. Run `adb devices` and approve USB debugging inside the headset."
        )

    def get_device(self) -> object:
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        return self.get_usb_device(client)

    def install(self, apk_path: str | None = None, verbose: bool = True, reinstall: bool = False) -> None:
        installed = self.device.is_installed(self.apk_name)
        if installed and not reinstall:
            if verbose:
                print("Oculus teleop APK is already installed.")
            return

        if apk_path is None:
            apk_path = str(Path(__file__).with_name("APK").joinpath("teleop-debug.apk"))
        if not Path(apk_path).is_file():
            raise FileNotFoundError(
                "Oculus teleop APK is not bundled in this repo. Install it manually from your existing "
                "teleop project or pass a valid APK path before using Oculus teleoperation."
            )

        success = self.device.install(apk_path, test=True, reinstall=reinstall)
        if not success or not self.device.is_installed(self.apk_name):
            raise RuntimeError("Failed to install Oculus teleop APK on the Quest headset.")

    @staticmethod
    def process_data(string: str) -> tuple[dict[str, np.ndarray], dict[str, bool | tuple[float, ...]]]:
        try:
            transforms_string, buttons_string = string.split("&")
        except ValueError:
            return {}, {}
        split_transform_strings = transforms_string.split("|")
        transforms: dict[str, np.ndarray] = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4, 4))
            pair = pair_string.split(":")
            if len(pair) != 2:
                continue
            left_right_char = pair[0]
            transform_string = pair[1]
            values = transform_string.split(" ")
            col = 0
            row = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[row][col] = float(value)
                col += 1
                if col >= 4:
                    col = 0
                    row += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        return transforms, parse_buttons(buttons_string)

    def extract_data(self, line: str) -> str:
        if self.tag not in line:
            return ""
        try:
            return line.split(self.tag + ": ")[1]
        except ValueError:
            return ""

    def get_transformations_and_buttons(self) -> tuple[dict[str, np.ndarray], dict[str, bool | tuple[float, ...]]]:
        with self._lock:
            return self.last_transforms, self.last_buttons

    def read_logcat_by_line(self, connection: object) -> None:
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if not data:
                    continue
                transforms, buttons = self.process_data(data)
                if not transforms and not buttons:
                    continue
                with self._lock:
                    self.last_transforms = transforms
                    self.last_buttons = buttons
                if self.fps_counter is not None:
                    self.fps_counter.get_and_print_fps()
            except UnicodeDecodeError:
                continue
        file_obj.close()
        connection.close()
