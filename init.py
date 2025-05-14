import time
import random
import sys
import os

def slow_type(text, speed=0.005):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(speed)
    print()

def progress(task, duration=1/2):
    bar = "[------------------------------]"
    sys.stdout.write(f"{task} {bar}")
    sys.stdout.flush()
    for i in range(30):
        time.sleep(duration / 50)
        sys.stdout.write("\r" + task + " [" + "#" * (i+1) + "-" * (29 - i) + "]")
        sys.stdout.flush()
    print(" ✔")

def fake_hex_dump(lines=8):
    for _ in range(lines):
        addr = hex(random.randint(0x10000000, 0xFFFFFFF0))
        data = " ".join(f"{random.randint(0, 255):02X}" for _ in range(16))
        slow_type(f"{addr}  {data}")

def spoof_serials():
    serials = {
        "BIOS": f"{random.randint(1000000000,9999999999)}",
        "MAC": f"00:{random.randint(10,99):02X}:{random.randint(10,99):02X}:{random.randint(10,99):02X}:{random.randint(10,99):02X}:{random.randint(10,99):02X}",
        "Drive": f"WDC-WX{random.randint(10000000,99999999)}"
    }
    for k, v in serials.items():
        slow_type(f"↻ Spoofed {k} Serial: {v}")
        time.sleep(0.3)

def main():
    os.system("clear" if os.name == "posix" else "cls")
    slow_type("Starting WHAR_LEWTS strikes back.")
    slow_type("🔧 Initializing PCIe DMA Interface - Mode: Stealth Undetected v3.7")
    time.sleep(0.5)
    progress("🔌 Enumerating PCI Devices")
    slow_type("🧬 Found: [8086:15B8] Intel DMA Controller")
    time.sleep(0.3)
    slow_type("📥 Mapping physical memory...")
    fake_hex_dump()
    time.sleep(0.5)

    progress("🛡️ Bypassing kernel integrity checks")
    slow_type("✔️ Kernel callback hooks cleared")
    progress("🧪 Injecting undetectable ring0 payload")
    time.sleep(0.5)

    slow_type("🔒 Authenticating to Fortnite anti-cheat driver...")
    slow_type("❗ Kernel driver rejected. Spoofing hardware identifiers...")
    spoof_serials()

    time.sleep(0.4)
    progress("📦 Streaming ESP, AimAssist, Radar Overlay")
    slow_type("📈 DMA sync at 1200MB/s -- Injection Stable")
    slow_type("✅ Fortnite DMA Injection Complete")
    slow_type("🖥️ Overlay running on monitor 1 - Ready to drop into Tilted Towers 🪂")
    progress("Enabling modules...")
    time.sleep(1)
    slow_type("🏃 Movement mode enabled. ( Webby Module )")
    time.sleep(1)
    slow_type("🎯 AIM Assist enabled. ( WhooHub Module )")
    time.sleep(1)
    slow_type("😎 Coolness mode enabled. ( Obeyfearless1 & DSPRING Module )")
    time.sleep(1)
    slow_type("😡 Crashout mode enabled. ( DiamondMace Module )")
    time.sleep(100)
if __name__ == "__main__":
    main()

