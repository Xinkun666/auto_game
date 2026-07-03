#!/usr/bin/env python3
# coding=utf-8

#
# Copyright (c) 2020-2022 Huawei Device Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import stat
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "grpcio",
    "protobuf",
]


def get_info(file_path):
    ver = "0.0.0"
    try:
        ver_file_open = os.open(file_path,
                                os.O_RDWR, stat.S_IWUSR | stat.S_IRUSR)
        with os.fdopen(ver_file_open, "r") as file_desc:
            lines = file_desc.readlines()
            for line in lines:
                if line.strip().startswith("VERSION"):
                    ver = line.split("=")[1].strip()
                    ver = ver.replace("'", "").replace('"', "")
    except Exception as e:
        ver = "0.0.0"
    return ver


def find_resources(exclude: list):
    """查询项目下所有目录 exclude：排除目录列表"""
    current_path = os.getcwd()
    all_item = os.listdir(current_path)
    result = []
    for item in all_item:
        item_path = os.path.join(current_path, item)
        if os.path.isdir(item_path):
            relative_path = os.path.relpath(item_path, current_path)
            relative_path = relative_path.replace("\\", "/")
            if relative_path not in exclude:
                result.append(relative_path+r"/*")
                all_item.extend([os.path.join(relative_path, x) for x in os.listdir(item_path)])
    return result


# 自动获取py文件目录列表
packages = find_packages()
packages = ["hoscrcpy_sdk." + x for x in packages]
packages.append("hoscrcpy_sdk")
print("packages", packages)
# 自动获取静态资源文件目录列表
exclude=["communication/proto", ".idea", "__pycache__", "dist"]
package_data_list = find_resources(exclude=exclude)
print("package_data_list", package_data_list)

version = get_info("__init__.py")
setup(
    name='hoscrcpy_sdk',
    description='hoscrcpy_sdk',
    version=version,
    url='',
    package_dir={'hoscrcpy_sdk': ''},
    packages=packages,
    package_data={
        "hoscrcpy_sdk": package_data_list
    },
    data_files=[],
    entry_points={

    },
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "full": []
    },
)
