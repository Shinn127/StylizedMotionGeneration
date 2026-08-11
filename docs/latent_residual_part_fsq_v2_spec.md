# Latent Residual Part-FSQ V2

V2 is an independent implementation of the latent residual part tokenizer. The
V1 module and checkpoint family remain available for direct comparison.

## Contract

- family: `latent_residual_fsq_v2`
- variant: `v2`
- representation id: `latent_residual_fsq_v2_40x9`
- architecture version: `3`
- coordinates: `base=20`, `torso=6`, `left_leg=4`, `right_leg=4`, `left_arm=3`, `right_arm=3`
- levels: `9`
- inference decoder passes: `1`
- batch size: `512`

## Data Flow

```text
x -> holistic causal encoder -> h_base -> Base-FSQ -> q_base

x_part -> shared causal part encoder -> u_part
q_base.detach() -> part predictor -> predicted_part
u_part - predicted_part -> Part-FSQ -> q_part
q_part -> dense full-latent projector -> delta_z_part

z = q_base + sum(delta_z_part)
z -> one shared causal decoder -> reconstruction
```

`q_base` and `z` are ordinary dense `[B,T,128]` decoder latents. V2 does not
use fixed part channel slices, concatenated structured decoder latents, padding,
or an orthogonal basis. Each part projector maps its 64-dimensional quantized
residual state directly to all base latent channels.

The additive accumulator is intentionally explicit. It preserves residual
composability and allows one part correction to be removed and replaced without
recomputing the other corrections.

## Part Editing

Part residuals are conditioned on the source base, so direct residual swapping
is invalid. For target `t`, donor `d`, and part `p`, V2 reconstructs the donor
part state and re-expresses it relative to the target base:

```text
donor_state = predictor_p(q_base_d) + q_part_d
target_residual = donor_state - predictor_p(q_base_t)
z_edit = z_target - delta_z_p_target + projector_p(target_residual)
```

This edited latent is decoded once. Training adds a simple masked transfer loss
for the selected part and a masked preserve loss for all other motion features.
The base reconstruction term is low-weight supervision; latent energy
regularization is disabled for V2 because it conflicts with usable edits.

## Compatibility

V2 uses a separate registry family, checkpoint metadata, config, and output
directory. V1 checkpoints are not loadable as V2 checkpoints and V2 does not
modify the V1 module.
