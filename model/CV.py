import subprocess
import sys
import os


def run_script(script_path):
    """
    运行子脚本并实时打印输出 (Real-time Output)
    这样可以防止在特征提取等耗时操作时控制台看起来像死机了一样
    """
    print(f"\n===> Executing: {script_path}")

    # 确保路径格式正确
    script_path = os.path.normpath(script_path)

    # 使用 Popen 进行流式输出
    # bufsize=1 表示行缓冲，text=True 表示文本模式
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 将错误输出合并到标准输出
        text=True,
        bufsize=1
    )

    # 实时逐行打印输出
    for line in process.stdout:
        print(f"[{os.path.basename(script_path)}] {line}", end='')

    # 等待进程结束
    process.wait()

    if process.returncode == 0:
        print(f"\n===> Successfully finished: {script_path}")
    else:
        print(f"\n===> Error in {script_path} with return code {process.returncode}")
        sys.exit(1)


def pipeline_controller():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # 1. 检查必要目录
    dirs = ['./svd_feature', './nonlinear_feature']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # 2. 阶段一：调用 SVD 特征提取
    svd_script = os.path.join('svd_feature', 'extract_svd_features.py')
    if os.path.exists(svd_script):
        run_script(svd_script)
    else:
        print(f"Warning: SVD extraction script not found at {svd_script}!")

    # 3. 阶段二：调用 动态超图 (DyH) 特征提取
    dyh_script = os.path.join('nonlinear_feature', 'extract_nonlinear_features.py')
    if os.path.exists(dyh_script):
        run_script(dyh_script)
    else:
        print(f"Warning: Nonlinear extraction script not found at {dyh_script}!")

    # 4. 阶段三：启动主程序进行评估
    print("\n" + "=" * 30)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 30)

    # 检查 main.py 是否存在
    if os.path.exists('main.py'):
        run_script('main.py')
    else:
        print("Error: main.py not found in current directory!")


if __name__ == "__main__":
    pipeline_controller()
