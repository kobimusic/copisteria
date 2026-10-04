# models

| File | What it is |
|---|---|
| `yolo_v7_taxonomy.json` | The v7 detector's 270 classes and its attribute vocabularies (staff position, stem, dots, voice); the evidence model's class head shares its index |

The weights are not in the repository. Put them here:

| File | Where it comes from |
|---|---|
| `evidence-2m.pt` | The evidence model (2.03M parameters) |
| `small/v7_obj-recall-30m_best.pt` | The v7 symbol detector (28.7M parameters), the default |
| `small/v7_obj-recall_best.pt` | The small v7 detector (1.94M parameters, the same 270 classes), for `--detector small` |

Both are in the Hugging Face repo `kobimusic/copista-evidence`.
