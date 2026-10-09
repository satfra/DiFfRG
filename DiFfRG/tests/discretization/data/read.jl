using HDF5

h5open("/tmp/DiFfRG_test_hdf5_output.h5", "r") do file
    ms = file["maps"]
    m_names = keys(file["maps"])
    global maps = Dict(m => Dict(
        "data" => ([read(step["data"]) for step in ms[m]]),
        "coords" => ([stack(read(step["coordinates"])) for step in ms[m]]),
        "times" => vcat([attrs(step)["time"] for step in ms[m]])
    ) for m in m_names)

    co = file["coordinates"]
    c_names = keys(file["coordinates"])
    global coords = Dict( n => stack(read(co[n])) for n in c_names)
end


maps["spline"]["coords"][1]
maps["spline"]["times"]
maps["spline"]["data"]


using 