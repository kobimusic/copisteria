# models

| File | What it is |
|---|---|
| `yolo_v7_taxonomy.json` | The v7 detector's 270 classes and its attribute vocabularies (staff position, stem, dots, voice); the evidence model's class head shares its index |

The weights are not in the repository. Put them here:

| File | Where it comes from |
|---|---|
| `evidence-2m.pt` | The evidence model (2.03M parameters), both sizes |
| `small/v7_obj-recall-30m_best.pt` | copista-28m's symbol detector (28.7M parameters), the default |
| `small/v7_obj-recall_best.pt` | copista-2m's symbol detector (1.94M parameters, the same 270 classes), `--detector 2m` |

All three are in the Hugging Face repo `kobimusic/copista-evidence`.
