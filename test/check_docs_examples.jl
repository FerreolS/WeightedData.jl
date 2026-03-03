using WeightedData
using StatsAPI
using Statistics

const DOC_INDEX = joinpath(@__DIR__, "..", "docs", "src", "index.md")
const EXCLUDED_BLOCK_PATTERNS = [
    r"\busing\s+CUDA\b",
]

function extract_julia_blocks(path::AbstractString)
    lines = readlines(path)
    in_block = false
    lang = ""
    blocks = NamedTuple{(:lang, :code, :index), Tuple{String, String, Int}}[]
    current = IOBuffer()
    block_index = 0

    for line in lines
        if startswith(line, "```")
            fence = strip(line)
            if !in_block
                lang = strip(replace(fence, "```" => ""))
                in_block = true
                current = IOBuffer()
                block_index += 1
            else
                code = String(take!(current))
                if lang == "julia" || lang == "julia-repl"
                    push!(blocks, (lang = lang, code = code, index = block_index))
                end
                in_block = false
                lang = ""
            end
            continue
        end

        in_block && println(current, line)
    end

    return blocks
end

function repl_to_code(block::AbstractString)
    code = IOBuffer()
    for line in split(block, '\n')
        startswith(line, "julia> ") && println(code, line[8:end])
    end
    return String(take!(code))
end

function should_skip(code::AbstractString)
    return any(p -> occursin(p, code), EXCLUDED_BLOCK_PATTERNS)
end

function main()
    blocks = extract_julia_blocks(DOC_INDEX)
    isempty(blocks) && error("No Julia/JULIA-REPL blocks found in docs/src/index.md")

    executed = 0
    skipped = 0
    for block in blocks
        raw_code = block.lang == "julia-repl" ? repl_to_code(block.code) : block.code
        isempty(strip(raw_code)) && continue

        if should_skip(raw_code)
            skipped += 1
            continue
        end

        try
            Base.include_string(Main, raw_code, "docs:index:block$(block.index)")
            executed += 1
        catch err
            @error "Docs example block failed" block_index = block.index language = block.lang exception = (err, catch_backtrace())
            rethrow()
        end
    end

    return println("Docs examples: OK (", executed, " block(s) executed, ", skipped, " skipped)")
end

main()
