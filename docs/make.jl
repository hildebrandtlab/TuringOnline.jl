using Documenter
include("../src/TuringOnline.jl")
import .online as online

Documenter.makedocs(
  root = "../",
  source = "src",
  build = "build",
  modules = Module[online],
  clean = true,
  doctest = true,
  pages = ["Index" => "index.md"],
  sitename = "TuringOnline.jl"
)

deploydocs(
    repo = "github.com/KonstantinBob/TuringOnline.jl.git",
)