{
  description = "BlitzRemesher C++20 development and research";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/4c7870105e7f1fdf9c48688c8d7efc21abf0688a";
  outputs = { self, nixpkgs }:
    let systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
    in { devShells = nixpkgs.lib.genAttrs systems (system:
      let pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in { default = (pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
        packages = with pkgs; [ cmake ninja pkg-config git jq gcc ];
        buildInputs = with pkgs; [ curl openssl nlohmann_json libarchive cgal boost eigen gmp mpfr ];
      };
      neural = (pkgs.mkShell.override { stdenv = pkgs.gcc14Stdenv; }) {
        packages = with pkgs; [ cmake ninja pkg-config git jq nodejs unzip patchelf ];
        buildInputs = with pkgs; [ curl openssl nlohmann_json libarchive cudaPackages.cudatoolkit ];
        CUDACXX = "${pkgs.cudaPackages.cudatoolkit}/bin/nvcc";
        CUDAHOSTCXX = "${pkgs.gcc14}/bin/g++";
        CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
        shellHook = ''
          export BLITZ_LIBTORCH_ROOT="$PWD/.cache/libtorch"
          export LD_LIBRARY_PATH="$BLITZ_LIBTORCH_ROOT/lib:/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH"
        '';
      }; }); };
}
