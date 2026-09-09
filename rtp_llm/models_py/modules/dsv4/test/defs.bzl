"""DSV4 test execution on the CUDA13 pools with an explicit pytest entrypoint."""

load("@arch_config//:arch_select.bzl", "cuda13_test_exec_properties")
load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")
load("@rules_python//python:defs.bzl", "py_test")

DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

def dsv4_py_test(name, deps, src = None, gpu_count = 1, tags = [], args = [], **kwargs):
    source = src or name + ".py"
    runner = "//rtp_llm/test/pytest:pytest_main.py"
    py_test(
        name = name,
        srcs = [source, runner],
        main = runner,
        args = ["--cuda-devices", str(gpu_count), "$(location :" + source + ")"] + args,
        deps = deps + [test_requirement("pytest")],
        tags = ["dsv4_cuda13"] + tags,
        env = {"GPU_COUNT": str(gpu_count)},
        exec_properties = cuda13_test_exec_properties(gpu_count),
        target_compatible_with = DSV4_CUDA13_ONLY,
        legacy_create_init = 0,
        **kwargs
    )
