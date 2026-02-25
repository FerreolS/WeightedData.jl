using Uncertain
using WeightedData
import WeightedData: get_value, get_precision, weightedmean

@testset "WeightedDataUncertainExt" begin
    @testset "WeightedValue from Uncertain.Value" begin
        # Test basic conversion from Uncertain.Value to WeightedValue
        u = Uncertain.Value(2.0, 0.5)
        w = WeightedValue(u)
        
        @test get_value(w) == 2.0
        @test get_precision(w) ≈ 1 / 0.5^2  # precision = uncertainty^(-2)
        @test get_precision(w) ≈ 4.0
    end
    
    @testset "Uncertain.Value from WeightedValue" begin
        # Test basic conversion from WeightedValue to Uncertain.Value
        w = WeightedValue(3.0, 16.0)  # precision = 16 => uncertainty = 1/sqrt(16) = 0.25
        u = Uncertain.Value(w)
        
        @test Uncertain.value(u) == 3.0
        @test Uncertain.uncertainty(u) ≈ 0.25
    end
    
    @testset "Round-trip conversion" begin
        # Verify that round-trip conversion preserves value and uncertainty
        u_orig = Uncertain.Value(1.5, 0.2)
        w = WeightedValue(u_orig)
        u_round = Uncertain.Value(w)
        
        @test Uncertain.value(u_round) ≈ Uncertain.value(u_orig)
        @test Uncertain.uncertainty(u_round) ≈ Uncertain.uncertainty(u_orig) atol=1e-15
    end
    
    @testset "Array conversions" begin
        # Test conversion of uncertain arrays
        u_array = [Uncertain.Value(1.0, 0.1), Uncertain.Value(2.0, 0.2), Uncertain.Value(3.0, 0.3)]
        w_array = WeightedValue.(u_array)
        
        @test length(w_array) == 3
        @test get_value(w_array[1]) == 1.0
        @test get_precision(w_array[1]) ≈ 100.0
        @test get_precision(w_array[2]) ≈ 25.0
        @test get_precision(w_array[3]) ≈ 100/9
    end
    
    @testset "High and low precision values" begin
        # Test with very small uncertainties (high precision)
        u_precise = Uncertain.Value(1.0, 1e-10)
        w_precise = WeightedValue(u_precise)
        @test get_precision(w_precise) ≈ 1e20
        
        # Test with very large uncertainties (low precision)
        u_imprecise = Uncertain.Value(1.0, 1e10)
        w_imprecise = WeightedValue(u_imprecise)
        @test get_precision(w_imprecise) ≈ 1e-20
    end
    
    @testset "Arithmetic with uncertain values" begin
        # Test that WeightedValue can be used in calculations with Uncertain values
        u1 = Uncertain.Value(1.0, 0.1)
        u2 = Uncertain.Value(2.0, 0.2)
        w1 = WeightedValue(u1)
        w2 = WeightedValue(u2)
        
        # Test arithmetic preserves structure
        w_sum = w1 + w2
        @test get_value(w_sum) ≈ 3.0
        
        # Convert back and verify uncertainty propagation
        u_back1 = Uncertain.Value(w1)
        u_back2 = Uncertain.Value(w2)
        
        @test Uncertain.value(u_back1) ≈ 1.0
        @test Uncertain.value(u_back2) ≈ 2.0
    end
    
    @testset "Zero uncertainty handling" begin
        # Test conversion with zero uncertainty (infinite precision)
        u_zero_unc = Uncertain.Value(5.0, 0.0)
        w_zero_unc = WeightedValue(u_zero_unc)
        
        @test get_value(w_zero_unc) == 5.0
        @test isinf(get_precision(w_zero_unc))
    end
    
    @testset "Negative value handling" begin
        # Test with negative values
        u_neg = Uncertain.Value(-3.0, 0.5)
        w_neg = WeightedValue(u_neg)
        
        @test get_value(w_neg) == -3.0
        @test get_precision(w_neg) ≈ 4.0
        
        u_back = Uncertain.Value(w_neg)
        @test Uncertain.value(u_back) ≈ -3.0
        @test Uncertain.uncertainty(u_back) ≈ 0.5
    end
    
    @testset "Type preservation" begin
        # Test that Float32 and Float64 types are handled
        u32 = Uncertain.Value(Float32(1.0), Float32(0.1))
        w32 = WeightedValue(u32)
        
        @test get_value(w32) ≈ 1.0f0 rtol=1e-6
        @test get_precision(w32) ≈ 100.0 rtol=1e-5
    end
    
    @testset "Weighted mean with uncertain values" begin
        # Test weighted mean of uncertain values converted to WeightedValues
        u1 = Uncertain.Value(1.0, 0.5)
        u2 = Uncertain.Value(3.0, 0.5)
        
        w1 = WeightedValue(u1)
        w2 = WeightedValue(u2)
        
        w_mean = weightedmean(w1, w2)
        @test get_value(w_mean) == 2.0
        @test get_precision(w_mean) ≈ 8.0  # 4 + 4
    end
end
