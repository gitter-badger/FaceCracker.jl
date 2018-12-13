# This file is a part of DeepToolKit. License is MIT

using Documenter, DeepToolKit

makedocs(
	modules = [DeepToolKit],
	sitename = "DeepToolKit.jl",
	pages = Any[
	  "index.md"
	],
	versions = ["v#.#", "dev" => "dev"],
	assets = [""],
)

deploydocs(
	repo = "github.com/fetaxyu/DeepToolKit.jl",
)
