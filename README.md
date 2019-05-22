# GAN-TUBerlin

## Thomas  Approach (T-Approach)
A train step consists of the following steps:
1. Sample a batch from the prior distribution, let's call it Z
2. Compute D(G(Z))
3. Take the set of Z_good which has the highest value in D(G(Z)), it can have 50% of the samples of Z
4. Do a regular GAN train step (train D, then train G) using Z_good
