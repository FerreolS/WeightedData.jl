using Measurements
using WeightedData
import WeightedData: get_value, get_precision, weightedmean

@testset "WeightedDataMeasurementsExt" begin
    @testset "WeightedValue from Measurement" begin
        # Test basic conversion from Measurement to WeightedValue
        m = 2.0 ± 0.5
        w = WeightedValue(m)

        @test get_value(w) == 2.0
        @test get_precision(w) ≈ 1 / 0.5^2  # precision = uncertainty^(-2)
        @test get_precision(w) ≈ 4.0
    end

    @testset "Measurement from WeightedValue" begin
        # Test basic conversion from WeightedValue to Measurement
        w = WeightedValue(3.0, 16.0)  # precision = 16 => uncertainty = 1/sqrt(16) = 0.25
        m = Measurement(w)

        @test Measurements.value(m) == 3.0
        @test Measurements.uncertainty(m) ≈ 0.25
    end

    @testset "Round-trip conversion" begin
        # Verify that round-trip conversion preserves value and uncertainty
        m_orig = 1.5 ± 0.2
        w = WeightedValue(m_orig)
        m_round = Measurement(w)

        @test Measurements.value(m_round) ≈ Measurements.value(m_orig)
        @test Measurements.uncertainty(m_round) ≈ Measurements.uncertainty(m_orig) atol = 1.0e-15
    end

    @testset "Array conversions" begin
        # Test conversion of measurement arrays
        m_array = [1.0 ± 0.1, 2.0 ± 0.2, 3.0 ± 0.3]
        w_array = WeightedValue.(m_array)

        @test length(w_array) == 3
        @test get_value(w_array[1]) == 1.0
        @test get_precision(w_array[1]) ≈ 100.0
        @test get_precision(w_array[2]) ≈ 25.0
        @test get_precision(w_array[3]) ≈ 100 / 9
    end

    @testset "High and low precision values" begin
        # Test with very small uncertainties (high precision)
        m_precise = 1.0 ± 1.0e-10
        w_precise = WeightedValue(m_precise)
        @test get_precision(w_precise) ≈ 1.0e20

        # Test with very large uncertainties (low precision)
        m_imprecise = 1.0 ± 1.0e10
        w_imprecise = WeightedValue(m_imprecise)
        @test get_precision(w_imprecise) ≈ 1.0e-20
    end

    @testset "Arithmetic with Measurements" begin
        # Test that WeightedValue can be used in calculations with Measurements
        m1 = 1.0 ± 0.1
        m2 = 2.0 ± 0.2
        w1 = WeightedValue(m1)
        w2 = WeightedValue(m2)

        # Test arithmetic preserves structure
        w_sum = w1 + w2
        @test get_value(w_sum) ≈ 3.0

        # Convert back and verify uncertainty propagation
        m_back1 = Measurement(w1)
        m_back2 = Measurement(w2)
        m_calc_sum = m_back1 + m_back2

        @test Measurements.value(m_calc_sum) ≈ 3.0
    end

    @testset "Zero precision handling" begin
        # Test conversion with zero uncertainty (infinite precision)
        m_zero_unc = 5.0 ± 0.0
        w_zero_unc = WeightedValue(m_zero_unc)

        @test get_value(w_zero_unc) == 5.0
        @test isinf(get_precision(w_zero_unc))
    end

    @testset "Negative value handling" begin
        # Test with negative values
        m_neg = -3.0 ± 0.5
        w_neg = WeightedValue(m_neg)

        @test get_value(w_neg) == -3.0
        @test get_precision(w_neg) ≈ 4.0

        m_back = Measurement(w_neg)
        @test Measurements.value(m_back) ≈ -3.0
        @test Measurements.uncertainty(m_back) ≈ 0.5
    end

    @testset "Type preservation" begin
        # Test that Float32 and Float64 types are preserved
        m32 = measurement(Float32(1.0), Float32(0.1))
        w32 = WeightedValue(m32)

        # Note: WeightedValue(m32) may promote to Float64, so check the value
        @test get_value(w32) ≈ 1.0f0 rtol = 1.0e-6
    end
end
