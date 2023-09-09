# { pkgs ? import <nixpkgs> {} }:
# pkgs.poetry2nix.mkPeotryApplication {
#   projectDir = ./.;
# }
let 
  overlay = self: super: {
    poetry2nix = super.poetry2nix.mkPeotryApplication {
      projectDir = ./.;
    };
  };
  hostPkgs = import <nixpkgs> { overlays = [ overlay ]; };
  linuxPkgs = import <nixpkgs> { overlays = [ overlay ]; system = "x86_64-linux"; };
in

{
  inherit hostPkgs linuxPkgs;
  docker  = hostPkgs.dockerTools.buildLayeredImage {
    name = "my-image";
    tag = "latest";
    config.Cmd = ["streamlit"  "run"  "--server.headless=true"  "--server.port=8501"  "st_multi_batch.py"];
  };

}
