import time
import sys
import os
from colorama import init, Fore, Style

init()

# Speed of slow typing
TYPE_SPEED = 0.0015

# Optional delay between sections
SECTION_DELAY = 0.015

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def slow_type(text, color=Fore.GREEN, newline=True):
    for char in text:
        sys.stdout.write(color + char + Style.RESET_ALL)
        sys.stdout.flush()
        time.sleep(TYPE_SPEED)
    if newline:
        print()

def pause(t=SECTION_DELAY):
    time.sleep(t)

def main():
    clear()
    banner = r"""
 __      __  ___ ___    _____ __________ 
/  \    /  \/   |   \  /  _  \\______   \
\   \/\/   /    ~    \/  /_\  \|       _/
 \        /\    Y    /    |    \    |   \
  \__/\  /  \___|_  /\____|__  /____|_  /
       \/         \/         \/       \/ 
.____     _____________      ______________________
|    |    \_   _____/  \    /  \__    ___/   _____/
|    |     |    __)_\   \/\/   / |    |  \_____  \ 
|    |___  |        \\        /  |    |  /        \
|_______ \/_______  / \__/\  /   |____| /_______  /
        \/        \/       \/                   \/ 
    """
    slow_type(banner, Fore.MAGENTA)
#    print(Fore.MAGENTA + banner + Style.RESET_ALL)
    pause()

    slow_type("🔧  SYSTEM SETUP", Fore.CYAN)
    slow_type("CPU: AMD Ryzen 7 7800X3D")
    slow_type("GPU: NVIDIA GeForce RTX 3090")
    slow_type("Monitor: LG ULTRAGEAR+ 240hz/4k (GSM780F)") 
    slow_type("Memory: 32GB DDR5-4800")
    slow_type("Mobo: MSI PRO B650-P WIFI (MS-7D78)")
    slow_type("NVME: Raid0 w/ 3x Samsung 990 Pro")
    slow_type("Case: Phanteks NV7 w/ Phanteks fans")
    slow_type("PSU: Antec 1300W Signature Platinum")

    pause()
    print()
    slow_type("⌨️  PERIPHERALS", Fore.CYAN)
    slow_type("Keyboard: Wooting 80HE ( Zinc Plate )")
    slow_type("Mouse: Cherry XTRFY MZ1 Wireless")
    slow_type("Mousepad: WALLHACK SP-004 ( Glass )")

    pause()
    print()
    slow_type("🎧  AUDIO & VIDEO", Fore.CYAN)
    slow_type("Headphones: Focal Celestee")
    slow_type("Speakers: Logitech z623")
    slow_type("Audio DAC: IFI hip-dac")
    slow_type("Mic: Shure MV6")
    slow_type("Webcam: Nikon D750 via Elgato HD60s+")

    print()
    slow_type(">> boot sequence complete.", Fore.YELLOW)

    time.sleep(100)
if __name__ == "__main__":
    main()

