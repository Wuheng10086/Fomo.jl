using Documenter
using Fomo

makedocs(
    sitename = "Fomo.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [Fomo],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/Wuheng10086/Fomo.jl.git",
    devbranch = "main",
)
