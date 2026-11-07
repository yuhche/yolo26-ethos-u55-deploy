"""
Probe and invoke a Grove Vision/SenseCraft device using python-sscma.

Usage:
    python scripts/sscma_probe.py --port /dev/ttyACM0 --baudrate 921600
    python scripts/sscma_probe.py --port /dev/ttyACM0 --invoke 1 --tscore 60 --tiou 50

Requires:
    pip install git+https://github.com/Seeed-Studio/python-sscma.git pyserial
"""

import argparse
import json
import time


def main():
    from sscma.micro.client import SerialClient
    from sscma.micro.device import Device

    parser = argparse.ArgumentParser(description="SSCMA serial probe/invoke helper")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--invoke", type=int, default=0, help="Invoke count. Use -1 for continuous.")
    parser.add_argument("--sample", type=int, default=0, help="Sample count. Use -1 for continuous.")
    parser.add_argument("--tscore", type=int, default=None, help="Score threshold 0..100")
    parser.add_argument("--tiou", type=int, default=None, help="IoU threshold 0..100")
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to monitor events")
    args = parser.parse_args()

    client = SerialClient(args.port, args.baudrate)
    device = Device(client)

    def on_connect(dev):
        print("Connected")
        print("Device:", dev.info.raw if dev.info else None)
        print("Model:", dev.model.raw if dev.model else None)

        if args.tscore is not None:
            dev.tscore = args.tscore
            print("Set TSCORE", args.tscore)
        if args.tiou is not None:
            dev.tiou = args.tiou
            print("Set TIOU", args.tiou)

        if args.sample:
            dev.Sample(args.sample)
        elif args.invoke:
            dev.Invoke(args.invoke, filter=False, show=True)

    def on_disconnect(_dev):
        print("Disconnected")

    def on_monitor(_dev, msg):
        printable = dict(msg)
        if "image" in printable:
            printable["image"] = f"<base64 jpeg {len(printable['image'])} chars>"
        print(json.dumps(printable, indent=2))

    def on_log(_dev, log):
        print("LOG:", log)

    device.on_connect = on_connect
    device.on_disconnect = on_disconnect
    device.on_monitor = on_monitor
    device.on_log = on_log

    print("Waiting for SSCMA device...")
    device.loop_start()
    try:
        time.sleep(args.seconds)
    finally:
        device.loop_stop()


if __name__ == "__main__":
    main()
