#!/usr/bin/env python3
"""Smoke-test a Linux ICD through the system Vulkan loader, without drawing."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import tempfile


class InstanceCreateInfo(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
                ("pApplicationInfo", c.c_void_p), ("enabledLayerCount", c.c_uint32),
                ("ppEnabledLayerNames", c.c_void_p), ("enabledExtensionCount", c.c_uint32),
                ("ppEnabledExtensionNames", c.c_void_p)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sgfx-icd-probe-") as directory:
        manifest = Path(directory) / "sgfx.json"
        manifest.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
            "library_path": str(args.library.resolve(strict=True)), "api_version": "1.0.0"}}))
        os.environ["VK_DRIVER_FILES"] = os.environ["VK_ICD_FILENAMES"] = str(manifest)
        loader = c.CDLL("libvulkan.so.1")
        loader.vkCreateInstance.argtypes = [c.POINTER(InstanceCreateInfo), c.c_void_p, c.POINTER(c.c_void_p)]
        loader.vkCreateInstance.restype = c.c_int32
        loader.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
        loader.vkDestroyInstance.restype = None
        loader.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.c_void_p]
        loader.vkEnumeratePhysicalDevices.restype = c.c_int32
        instance = c.c_void_p()
        status = loader.vkCreateInstance(c.byref(InstanceCreateInfo(sType=1)), None, c.byref(instance))
        if status != 0:
            raise RuntimeError(f"vkCreateInstance failed: {status}")
        try:
            count = c.c_uint32()
            status = loader.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
            # The system loader may return INITIALIZATION_FAILED when no ICD
            # reports a GPU. This is expected on a Linux build host without Scarlet.
            if status not in (0, -3) or (status == -3 and count.value != 0):
                raise RuntimeError(f"vkEnumeratePhysicalDevices failed: {status}")
            print(f"PASS: system Vulkan loader created an ICD instance; adapters={count.value}, GPU execution untested")
        finally:
            loader.vkDestroyInstance(instance, None)


if __name__ == "__main__":
    main()
