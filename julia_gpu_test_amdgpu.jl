#!/usr/bin/env julia
"""
julia_gpu_test_amdgpu.jl

Simple AMDGPU.jl GPU test script.

How device selection works:
  - This script DOES NOT select devices.
  - To choose a specific GPU, set HIP_VISIBLE_DEVICES before running.
    HIP uses 0-based indexing over the GPUs it sees.

Examples:
  # List devices + run tests on default device selection:
  julia --project=. ./julia_gpu_test_amdgpu.jl

  # Force GPU 0 only:
  HIP_VISIBLE_DEVICES=0 julia --project=. ./julia_gpu_test_amdgpu.jl

  # Force GPU 1 only:
  HIP_VISIBLE_DEVICES=1 julia --project=. ./julia_gpu_test_amdgpu.jl

Optional (useful for some iGPUs, e.g. Radeon 780M / gfx1103):
  HSA_OVERRIDE_GFX_VERSION=11.0.0 julia --project=. ./julia_gpu_test_amdgpu.jl
"""

using AMDGPU
using LinearAlgebra
using Statistics

banner(msg) = (println("\n", "="^60); println(msg); println("="^60))
ok(msg) = println("OK ", msg)

function fail(msg, e)
    println("FAILED..... ", msg)
    showerror(stdout, e, catch_backtrace())
    println()
end

function main()::Int
    banner("AMDGPU version info + devices")
    try
        AMDGPU.versioninfo()
        ok("Runtime initialized")
    catch e
        fail("AMDGPU.versioninfo() failed", e)
        return 1
    end

    banner("Active GPU (device being tested)")
    try
        devs = AMDGPU.devices()
        if isempty(devs)
            println("No AMDGPU devices found.")
            return 1
        end

        d0 = devs[1]  # HIP device 0 (first visible device)
        println("HIP_VISIBLE_DEVICES = ", get(ENV, "HIP_VISIBLE_DEVICES", "(not set; all visible)"))
        println("Testing HIP device 0 => ", d0)
        ok("Active GPU printed")
    catch e
        fail("Could not determine active GPU", e)
        return 1
    end


    banner("Basic allocation")
    try
        AMDGPU.zeros(Float32, 1024)
        AMDGPU.ones(Float32, 1024)
        ok("zeros / ones allocation works")
    catch e
        fail("Device allocation failed", e)
        return 1
    end

    banner("Host ↔ Device copy")
    try
        h  = collect(Float32, 1:1024)
        d  = AMDGPU.ROCArray(h)   # host -> device
        h2 = Array(d)             # device -> host
        @assert h == h2
        ok("Host <-> device copy works")
    catch e
        fail("Memory copy failed", e)
        return 1
    end

    banner("Simple computation")
    try
        A = AMDGPU.ones(Float32, 1024)
        B = 2f0 .* A
        s = sum(B)
        @assert isapprox(s, 2048f0)
        ok("Kernel execution works")
    catch e
        fail("Kernel execution failed", e)
        return 1
    end

    banner("Random number generation")
    try
        R = AMDGPU.rand(Float32, 1024)
        m = mean(R)
        ok("AMDGPU.rand works (mean = $(m))")
    catch e
        fail("AMDGPU.rand failed (rocRAND/ROCm issue possible)", e)
        println("\nWorkarounds:")
        println("  • Try a different GPU via HIP_VISIBLE_DEVICES (0, 1, ...)")
        println("  • Bypass rocRAND: CPU rand + copy to GPU:")
        println("      julia -e 'using AMDGPU; x=rand(Float32,1024); d=AMDGPU.ROCArray(x); println(sum(d))'")
        println("  • Upgrade ROCm if you're on an older release.")
        return 1
    end

    banner("SUCCESS")
    println("WOOHOOO All AMDGPU tests passed.")
    return 0
end

exit(main())
