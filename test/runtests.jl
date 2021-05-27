# packages required for test
using Test
using LinearAlgebra
using Random
using ForwardDiff
using Zygote

# packages tested
using KFEstimate

@testset "KFEstimate.jl" begin
    # Write your tests here.
    testdir = joinpath(dirname(@__DIR__), "test")
    @time @testset "Test KF" begin
        include(joinpath(testdir, "test_kf.jl"))
    end
end
