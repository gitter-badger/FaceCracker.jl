"""
    is_odd(x::Int) -> Bool
"""
is_odd(x::Int) = x & 0x1 == 1


"""
    list_files(dir::AbstractString=".")
Return the files in the directory dir
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