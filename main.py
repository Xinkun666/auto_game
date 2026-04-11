import os
from xdevice.__main__ import main_process

# 这种方式是通过SP打开想要的游戏，进入游戏后再启动游戏自动化
# if __name__ == '__main__':
#
#     # main_process("run -l testcases/pubg_test/pubg_1") # 启动和平精英测试
#     # main_process("run -l testcases/sanjiaozhou/sanjiaozhou_changgongxigu") # 启动三角洲
#     main_process("run -l testcases/pubg/和平精英全流程/auto_pubg")


# 这种方式是直接开启游戏自动化，需要人工手动点开游戏
if __name__ == '__main__':

    project_case = 'Auto_PUBG_ALL'  # 这是你在标注工具导出的自动化资源目录名
    target_case = "auto_pubg"  # 这是你编写的自动化用例脚本名
    os.environ["TARGET_PROJECT_CASE"] = project_case
    os.environ["TARGET_GAME_CASE"] = target_case
    from aw.autogame.tools.GameAutomator import GameAutomator
    automator = GameAutomator(driver=None, logger=None)
    automator.start()