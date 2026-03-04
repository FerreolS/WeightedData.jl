module WeightedDataAdaptExt

import Adapt:adapt_structure, adapt
import WeightedData:WeightedArray, get_value, get_precision

"""
    adapt_structure(to, wd::WeightedArray)

Adapt a `WeightedArray` structure to a target backend using `Adapt.jl`.

Both value and precision arrays are adapted consistently and wrapped back into
`WeightedArray`.
"""
adapt_structure(to, wd::WeightedArray) = 
	WeightedArray(adapt(to, get_value(wd)),adapt(to, get_precision(wd)))
end