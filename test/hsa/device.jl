@testset "Devices" begin
    @testset "Device IDs" begin
        devices = AMDGPU.devices()
        for (idx,device) in enumerate(devices)
            @test AMDGPU.device_id(device) == idx
        end
    end

    @testset "Default selection" begin
        device = AMDGPU.default_device()
        @test device !== nothing
        @test device == AMDGPU.device()

        device_name = HIP.name(device)
        @test length(device_name) > 0
        @test !occursin('\0', device_name)

        if length(AMDGPU.devices()) > 1
            @testset "Multi-GPU" begin
                init_device = AMDGPU.default_device()
                init_device_id = AMDGPU.default_device_id()
                @test init_device_id == 1

                AMDGPU.default_device_id!(2)
                @test AMDGPU.default_device_id() == 2
                @test AMDGPU.default_device() != init_device

                AMDGPU.default_device_id!(1)
                @test AMDGPU.default_device_id() == 1
                @test AMDGPU.default_device() == init_device

                @test_throws BoundsError AMDGPU.default_device_id!(0)
                @test_throws BoundsError AMDGPU.default_device_id!(length(AMDGPU.devices())+1)
            end
        else
            @test_skip "Multi-GPU"
        end
    end

    @testset "ISAs" begin
        device = AMDGPU.default_device()
        device_isa, features = AMDGPU.default_isa(device).arch_features
        @test length(device_isa) > 0
        @test occursin("gfx", device_isa)
    end
end
