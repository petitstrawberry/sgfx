# Private test build of Scarlet's Cocoa/VirGL QEMU. The OpenGL renderer and
# view still exist, but no window is shown and the process cannot take focus.
# This changes no installed QEMU binary or Scarlet flake.
{ scarlet }:
let
  flake = builtins.getFlake ("git+file://" + scarlet);
  original = builtins.head (builtins.filter (p: (p.pname or "") == "qemu") (flake.devShells.aarch64-darwin.default.buildInputs ++ flake.devShells.aarch64-darwin.default.nativeBuildInputs));
in original.overrideAttrs (old: {
  pname = "qemu-sgfx-offscreen";
  configureFlags = builtins.filter (s: !(builtins.match "--target-list=.*" s != null)) old.configureFlags ++ [ "--target-list=aarch64-softmmu" ];
  postPatch = (old.postPatch or "") + ''
    substituteInPlace ui/cocoa.m \
      --replace-fail '[window makeKeyAndOrderFront:self];' '/* Offscreen SGFX diagnostic: do not show windows. */' \
      --replace-fail 'kProcessTransformToForegroundApplication' 'kProcessTransformToUIElementApplication' \
      --replace-fail '[QemuApplication sharedApplication];' '[QemuApplication sharedApplication]; [NSApp setActivationPolicy:NSApplicationActivationPolicyProhibited];'
  '';
})
