# Bias, variance and the data you train on

A model's error on new data has three parts: **bias** (it is too simple to capture the pattern), **variance** (it fits noise in the particular training sample) and irreducible noise. You diagnose them by comparing errors:

| Training error | Validation error | Diagnosis | Try |
|---|---|---|---|
| high | high, close to training | underfitting (bias) | a richer model, better features, train longer, less regularization |
| low | much higher | overfitting (variance) | more data, regularization, a simpler model, early stopping, augmentation |

<figure class="diagram"><svg viewBox="0 0 600 236" width="600" role="img" aria-label="Training and validation error against model complexity">
<path d="M60,20 L60,190 L580,190" class="d-line"/>
<path d="M60.0,37.0 L68.7,44.5 L77.3,51.6 L86.0,58.4 L94.7,64.8 L103.3,70.8 L112.0,76.6 L120.7,82.0 L129.3,87.2 L138.0,92.1 L146.7,96.7 L155.3,101.1 L164.0,105.3 L172.7,109.3 L181.3,113.0 L190.0,116.6 L198.7,119.9 L207.3,123.1 L216.0,126.2 L224.7,129.0 L233.3,131.8 L242.0,134.4 L250.7,136.8 L259.3,139.1 L268.0,141.3 L276.7,143.4 L285.3,145.4 L294.0,147.3 L302.7,149.0 L311.3,150.7 L320.0,152.3 L328.7,153.8 L337.3,155.3 L346.0,156.6 L354.7,157.9 L363.3,159.2 L372.0,160.3 L380.7,161.4 L389.3,162.5 L398.0,163.4 L406.7,164.4 L415.3,165.3 L424.0,166.1 L432.7,166.9 L441.3,167.7 L450.0,168.4 L458.7,169.1 L467.3,169.7 L476.0,170.3 L484.7,170.9 L493.3,171.5 L502.0,172.0 L510.7,172.5 L519.3,172.9 L528.0,173.4 L536.7,173.8 L545.3,174.2 L554.0,174.6 L562.7,174.9 L571.3,175.3 L580.0,175.6" class="d-line" style="stroke:var(--d-a-ink);stroke-width:2.5"/>
<path d="M60.0,28.5 L68.7,36.0 L77.3,43.1 L86.0,49.8 L94.7,56.1 L103.3,62.1 L112.0,67.8 L120.7,73.0 L129.3,78.0 L138.0,82.7 L146.7,87.0 L155.3,91.1 L164.0,94.9 L172.7,98.4 L181.3,101.6 L190.0,104.6 L198.7,107.3 L207.3,109.8 L216.0,112.1 L224.7,114.1 L233.3,115.9 L242.0,117.5 L250.7,118.9 L259.3,120.1 L268.0,121.1 L276.7,121.8 L285.3,122.4 L294.0,122.8 L302.7,123.0 L311.3,123.0 L320.0,122.8 L328.7,122.4 L337.3,121.9 L346.0,121.2 L354.7,120.3 L363.3,119.3 L372.0,118.0 L380.7,116.6 L389.3,115.1 L398.0,113.3 L406.7,111.5 L415.3,109.4 L424.0,107.2 L432.7,104.8 L441.3,102.2 L450.0,99.5 L458.7,96.7 L467.3,93.6 L476.0,90.5 L484.7,87.1 L493.3,83.6 L502.0,79.9 L510.7,76.1 L519.3,72.1 L528.0,67.9 L536.7,63.6 L545.3,59.1 L554.0,54.5 L562.7,49.7 L571.3,44.7 L580.0,39.6" class="d-line" style="stroke:var(--d-c-ink);stroke-width:2.5"/>
<line x1="311.33333333333337" y1="20" x2="311.33333333333337" y2="190" class="d-line d-dash"/>
<text x="317.33333333333337" y="34" class="d-muted" text-anchor="start">best complexity</text>
<text x="64" y="206" class="d-muted" text-anchor="start">← underfitting: both errors high</text>
<text x="580" y="206" class="d-muted" text-anchor="end">overfitting: the gap grows →</text>
<text x="574" y="57.60986152512609" class="d-ink-c d-text" text-anchor="end">validation error</text>
<text x="574" y="167.6098615251261" class="d-ink-a d-text" text-anchor="end">training error</text>
<text x="320.0" y="224" class="d-muted" text-anchor="middle">model complexity (or training time) →</text>
<text x="24" y="105.0" class="d-muted" text-anchor="middle" transform="rotate(-90 24 105.0)">error</text>
</svg>
<figcaption>Training error keeps falling as the model gets more flexible; validation error falls, then rises once the model starts fitting noise. Read the diagnosis table above off this picture: a gap means variance, two high curves mean bias.</figcaption></figure>

**Regularization** trades a little bias for less variance:

- **L2** (ridge, weight decay) adds $\lambda \|w\|_2^2$: shrinks all weights smoothly.
- **L1** (lasso) adds $\lambda \|w\|_1$: drives some weights exactly to zero, selecting features.
- **Dropout**, **early stopping** and **data augmentation** regularize networks.

**The data matters more than the model.** Problems interviewers like to probe:

- **Leakage:** a feature that encodes the label or comes from the future (a "refund issued" flag when predicting fraud), or preprocessing fit on the full data before splitting. Scores look great and fall apart in production.
- **Class imbalance:** resample (over-sample the minority, under-sample the majority), weight the loss, and pick metrics that notice the minority class.
- **Missing values:** impute (and add an "is missing" indicator, since missingness itself is often informative) or use models that handle them.
- **Distribution shift:** the data in production differs from training (new users, new season). Monitor input distributions and model scores after launch.
- **Sampling bias:** the training sample is not the population you care about.

> [!TIP]
> When asked "the model overfits, what do you do?", say first how you know (the gap between training and validation curves), then fixes in order of cost: more or cleaner data, regularization, a smaller model.
