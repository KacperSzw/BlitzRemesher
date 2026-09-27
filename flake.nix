{
  description = "BlitzRemesher C++20 development and research";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/4c7870105e7f1fdf9c48688c8d7efc21abf0688a";
  outputs = { self, nixpkgs }:
    let systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
    in { devShells = nixpkgs.lib.genAttrs systems (system:
      let pkgs = import nixpkgs { inherit system; };
      in { default = (pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
        packages = with pkgs; [ cmake ninja pkg-config git jq gcc ];
        buildInputs = with pkgs; [ curl openssl nlohmann_json libarchive cgal boost eigen gmp mpfr ];
      }; }); };
}
