# Memory bound softmax kernel in CuTe DSL

## Naive Algorithm

%% insert LaTeX of softmax formulation

%% go over arithmetic intensity of softmax and why computing is free bc accelator flops/byte is much higher
```py
# %% insert pytorch softmax here
```

%% insert naive CuTe DSL implementation 

## Online Safe Softmax

%% explain why safe softmax is necessary for numerical correctness

%% explain how online softmax works, add derivation 

## Backward Pass
%% insert derivation for backward pass

## Caching the values in registers
%% justify caching values 

## Vectorization and Rewriting it using CuTe's Layout Algebra

%% insert a basic explanation of CuTe's layout algebra

%% briefly go over vectorized memory loads
maybe add a table of largest vector load vs generation

%% add fun layout trick

## Benchmarking Methodology
%% justify thorough and careful benchmarking

%% mention high profile incidents like Sakana.ai 

%% explain why to be careful of relative speedup and why % roofline measurements can be better, also explain why generous baselines matter bc of sandbagging.

%% show why L2 cache clearing matter

%% show why cuda graph capture during benchmarking also matter

%% justify torch.copy reference

%% maybe show off benchmarking technique of running

%% address limitations eg adversarial gaming and why reward hacking necessitates even more thorough benchmarking harnesses

## Final Results 
%% results and justify adding a compile time online softmax

%% address limitations like other hardware, no cluster reduce, and very large softmaxes on low batch sizes.

## References

%% mention batch invariance and link thinking machiines blog post