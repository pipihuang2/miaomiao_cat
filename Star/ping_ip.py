import subprocess
import platform
import concurrent.futures
import re

def ping(ip):
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "1", ip]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        return ip, (result.returncode == 0)
    except subprocess.TimeoutExpired:
        return ip, False

def get_mac(ip):
    system = platform.system().lower()

    if "windows" in system:
        try:
            result = subprocess.check_output(["arp", "-a", ip], universal_newlines=True)
            mac = re.search(r"([0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}", result)
            return mac.group(0) if mac else None
        except subprocess.CalledProcessError:
            return None
    else:
        try:
            result = subprocess.check_output(["arp", "-n", ip], universal_newlines=True)
            mac = re.search(r"([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", result)
            return mac.group(0) if mac else None
        except subprocess.CalledProcessError:
            return None

def main():
    base_ip = "192.168.11."
    ip_range = range(1, 255)

    alive_ips = []

    print("开始扫描...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(ping, base_ip + str(i)): i for i in ip_range}
        for future in concurrent.futures.as_completed(futures):
            ip, is_alive = future.result()
            if is_alive:
                alive_ips.append(ip)

    print("\n获取MAC地址中...")

    for ip in alive_ips:
        mac = get_mac(ip)
        if mac:  # 只有找到 MAC 地址才打印
            print(f"{ip} --> MAC地址: {mac}")

if __name__ == "__main__":
    main()
