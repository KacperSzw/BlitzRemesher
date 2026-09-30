This saved, finite FP32 model/input reproduced the L40S failure at step
3,405,312. The old native and Torch GEMMs agreed exactly, but differed from the
independent FP64 oracle by 0.000232670 (limit: 0.000200). Large cancellation in
the second hidden layer amplified accumulation error. The fixture protects the
unchanged numerical contract; it is not a training-quality benchmark.

Source: collected, checksum-verified `runpod-checkpoint-forensics-02`, revision
`abdb75a`. `expected.pt` retains the failing old GPU output for diagnosis.
