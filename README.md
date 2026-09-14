# Retinal image classification study code

This is the cleaned, data-free code package for [“Generative Artificial Intelligence for Retinal Image Translation to Improve Glaucoma Screening With Deep Learning”](https://pubmed.ncbi.nlm.nih.gov/41268987/) (Majid, Zebardast, and Wang; *Translational Vision Science & Technology*, 2025). It reconstructs workflows from five research notebooks: SLO/CFP index matching, binary CFP classification, patient-level low-data experiments, CycleGAN-generated synthetic CFP augmentation, and borderline-label evaluation. The source notebooks identify the generated images as CycleGAN `fake_B` outputs; this repository uses those images only as private training inputs and does not include the CycleGAN training code or weights. It does **not** contain patient records, images, model weights, clinical tables, saved notebook outputs, or original storage locations.

## Private inputs

Keep all input CSVs, images, model weights, and run outputs outside the public repository. The `.gitignore` adds a second safeguard, but it cannot protect files already committed or renamed to an unrecognized extension. Review the final staged files before publishing.

The main manifest has these columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | Stable **private** patient identifier; never publish it |
| `sample_id` | Stable **private** eye or visit identifier used for the clinical label |
| `image_path` | Local path to an image or `.npz` file |
| `label` | `0` healthy, `1` glaucoma, `2` borderline (evaluation only) |
| `split` | `train`, `val`, or `test` |
| `image_type` | `cfp`, `slo`, or `synthetic_cfp` |

The validator rejects patients shared across splits, conflicting sample labels, duplicate images, and synthetic images outside training. `patient_id` must identify the person, even when the label is eye-specific; `sample_id` identifies the labeled eye or visit. Use an existing, approved patient-level split. For `.npz` input, the image must be under `colorfundus`; the manifest supplies the label.

To derive labels from a private clinical table, provide a CSV with `sample_id,md,ght,psd_p_value` and an image index with `patient_id,sample_id,image_path,split,image_type`. The definitions are from [Majid, Zebardast, and Wang, *Translational Vision Science & Technology* (2025)](https://pubmed.ncbi.nlm.nih.gov/41268987/): **glaucoma** requires MD < -3 dB, abnormal GHT, and PSD p-value <= 0.05; **non-glaucoma** requires MD >= -1 dB, normal GHT, and PSD p-value > 0.05. Cases meeting neither definition are labeled `2` (borderline/unclear). The code accepts `normal` or `1` and `abnormal` or `3` for GHT, matching the notebook's codes; verify this mapping against your private data dictionary. Missing or invalid clinical values and conflicting sample labels cause an error. Borderline/unclear samples are omitted from training and validation and retained in the test manifest for sensitivity analyses.

## Install and run

Use a Python environment with the dependencies in `requirements.txt`. TensorFlow installation depends on the training system. Run these commands from the repository root, replacing bracketed paths with private local paths:

```bash
python -m pip install -r requirements.txt
python -m retina_study.prepare --clinical <private-clinical.csv> --images <private-image-index.csv> --output <private-manifest.csv>
python -m retina_study.train --manifest <private-manifest.csv> --output-dir <private-run-directory> --backbone vgg19
python -m retina_study.evaluate --manifest <private-manifest.csv> --model <private-run-directory>/best_model.keras --output <private-metrics.json>
```

For low-data experiments, rerun training with `--fraction 0.2`, `0.4`, `0.6`, `0.8`, or `1.0`, using a separate output directory for each run. The same seed gives nested patient-level subsets; patients with a positive label in either eye are in the positive stratum. Add `--include-synthetic` to include synthetic CFP images belonging to selected training patients. Generate synthetic images only from training-source patients before adding them to the manifest. Validation and test use real CFP images only. To compare backbones, use `--backbone resnet101` or `efficientnetb0`.

For SLO/CFP matching, prepare two **private** indexes with `match_key,image_path`, where the key encodes the approved exact matching unit (such as patient, eye, and visit), then run:

```bash
python -m retina_study.match_pairs --slo-index <private-slo-index.csv> --cfp-index <private-cfp-index.csv> --output <private-pairs.csv>
```

The output pair index contains private identifiers and paths. Do not publish it. The matching script rejects ambiguous duplicate keys rather than silently choosing one record.
