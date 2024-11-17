function gen(path::String, nspin::Integer)
    open(path, "w") do io
        for i in 0:nspin-1
            for j in i+1:nspin-1
                println(io, "$i $j $(rand([-1, 1]))")
            end
        end
    end
end

gen("test/externel_std/100_example.txt", 100)