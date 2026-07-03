#!/usr/bin/env python3
# coding=utf-8
import argparse
import shutil
import subprocess
import os
python = ""


def install_pyarmor():
    """安装混淆工具"""

    cmd = [python_exe, "-m", "pip", "install", "pyarmor==9.0.7", "-i", r"http://mirrors.tools.huawei.com/pypi/simple/"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = p.communicate()
    print("-----------install pyarmor-------------")
    e = e.decode()
    print(e)
    o = o.decode()
    print(o)


def exec_obfuscate():
    """执行混淆"""
    install_pyarmor()
    path = os.path.dirname(__file__)
    print(path)
    if os.path.exists(os.path.join(path, "dist/hoscrcpy_sdk")):
        print("删除旧的混淆文件")
        shutil.rmtree(os.path.join(path, "dist/hoscrcpy_sdk"))
    pyarmor_path = os.path.join(python, "Scripts", "pyarmor.exe")
    cmd = [pyarmor_path, "gen", "-O", "dist", path, "--platform", "windows.x86_64", "--platform",
        "darwin.x86_64", "--platform", "linux.x86_64", "-i"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = p.communicate()
    print("-----------obfuscate-------------")
    e = e.decode("utf8", "ignore")
    print(e)
    o = o.decode()
    print(o)


def exec_setup(is_obfuscate=False):
    """执行打包"""
    if is_obfuscate:
        # 切换工作目录
        dist_path = os.path.join(os.path.dirname(__file__), "dist", "hoscrcpy_sdk")
        os.chdir(dist_path)
        cmd = [python_exe, r".\setup.py", "sdist"]
    else:
        cmd = [python_exe, r".\setup.py", "sdist"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = p.communicate()
    print("-----------setup-------------")
    e = e.decode()
    print(e)
    o = o.decode()
    print(o)


def main(is_obfuscate=False):
    if is_obfuscate:
        exec_obfuscate()
    exec_setup(is_obfuscate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--obfuscate",
                        action='store_true',
                        default=False,
                        help="obfuscate py")
    parser.add_argument("-p", "--python",
                        action='store',
                        default='',
                        help="python path")
    args = parser.parse_args()
    print(args.obfuscate)
    print(args.python)
    if args.python:
        python = args.python
        python_exe = os.path.join(python, "python.exe")
    else:
        raise Exception("please provide python path e.g:-p D:/python3.11.9/")
    main(args.obfuscate)
