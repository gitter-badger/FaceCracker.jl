using Dates

# Rata Die milliseconds for 1970-01-01T00:00:00
const RATAEPOCH = Dates.value(DateTime(1970))

"""
    is_odd(x::Int) -> Bool
"""
is_odd(x::Int) = x & 0x1 == 1


"""
    list_files(dir::AbstractString=".")
Returns the files in the directory dir
"""
function list_files(dir::AbstractString=".")
    rs = Vector{String}()
    for (root, dirs, files) in walkdir(dir)
        for file in files
            push!(rs, joinpath(root, file))
        end
    end
    rs
end

"""
    time_millis(dt::DateTime=now())
Returns millisecond since 1970-01-01T00:00:00
"""
time_millis(dt::DateTime=now()) = Dates.value(dt) - RATAEPOCH
