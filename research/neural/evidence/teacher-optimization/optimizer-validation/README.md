# Invalid optimizer input regressions

Two malformed-input bugs were reproduced locally and corrected by
`f822850` (`Reject invalid loss settings and negative Adam checkpoint variance`).
Neither reproduction used a trained model or the completed one-hour checkpoint.
There is no evidence here that either input occurred in that training run, and
these checks make no quality or performance claim.

The tiny GPU probe used the pre-fix source at
`b947db8f7932bfd61f2ecda43cc43efec2624c57`:

- A NaN ranking margin with zero auxiliary and penalty weights was accepted.
  CUDA returned zero loss and gradient with failure flag zero, while the
  reference ReLU result was NaN.
- An optimizer checkpoint containing finite variance `-1` was accepted. One
  zero-gradient update produced a nonfinite parameter, advanced the step to 1,
  and left the failure flag at zero.

The fix validates scalar settings before constructor allocation or any direct
loss/Adam kernel launch. Margin and both loss weights must be finite and
nonnegative; zero remains valid. Existing learning-rate, decay, and norm limits
are preserved. Checkpoint loading now rejects negative variance as well as
nonfinite tensors. The variance scan occurs only during loading. Rejection
remains nontransactional, matching the existing load contract; a rejected
optimizer must not be used for further updates.

The affected action-train target built successfully. Its focused GPU CTest
passed 1/1 with no skips, and Compute Sanitizer memcheck reported zero errors.
The suite checks invalid scalar values at every host launch entry, valid zero
loss coefficients against autograd, corrupt variance away from the first tensor
element, valid zero/positive saved variance, and existing exact graph-captured
checkpoint continuation. Valid update arithmetic was not changed.

[record.json](record.json) records tested source and binary identities, results,
and validation commands. Each tested source hash was checked against commit
`f822850`. [before.cpp](before.cpp) and [build-commands.json](build-commands.json)
preserve the reproduction and exact local compiler/linker commands. They use
the repository's pinned Nix CUDA/LibTorch environment; absolute build paths are
local provenance, not a portable prebuilt executable. No binary, model,
optimizer payload, or new training data is archived here.

[manifest.json](manifest.json) verifies archived bytes and the original bytes
of copied/compressed artifacts. The probe build retains its `_FORTIFY_SOURCE`
warning from compiling without optimization. The post-fix build log retains
existing compiler warnings as well as the successful link; no warning-free-build
claim is made.
