# This file is a part of FaceCraker. License is MIT

using Documenter, FaceCraker

makedocs(
	modules = [FaceCraker],
	sitename = "FaceCraker.jl",
	pages = Any[
	  "index.md"
	],
	versions = ["v#.#", "dev" => "dev"],
	assets = [""],
)

deploydocs(
	repo = "github.com/fetaxyu/FaceCraker.jl",
)
