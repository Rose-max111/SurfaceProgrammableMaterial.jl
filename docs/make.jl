using SurfaceProgrammableMaterial
using Documenter

DocMeta.setdocmeta!(SurfaceProgrammableMaterial, :DocTestSetup, :(using SurfaceProgrammableMaterial); recursive=true)

makedocs(;
    modules=[SurfaceProgrammableMaterial],
    authors="Rose_max111 <luyimingboy@163.com> and contributors",
    sitename="SurfaceProgrammableMaterial.jl",
    format=Documenter.HTML(;
        canonical="https://Rose_max111.github.io/SurfaceProgrammableMaterial.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Rose_max111/SurfaceProgrammableMaterial.jl",
    devbranch="main",
)
