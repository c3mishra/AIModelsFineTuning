# Custom User Image Data

Use this folder to provide per-user training images. The notebook will prefer this data over synthetic/profile data when present.

## Folder layout

```
projects/clothing-preference-model/data/custom/
  user_0/
    likes/
      # optional class subfolders matching Fashion-MNIST names
      # e.g., T-shirt/top/, Trouser/, Pullover/, Dress/, Coat/,
      #       Sandal/, Shirt/, Sneaker/, Bag/, Ankle boot/
    dislikes/
      # same optional class subfolders
  user_1/
    likes/
    dislikes/
  ...
```

## Notes
- User ID is taken from the folder name: `user_0` -> 0, `user_12` -> 12.
- If you place images directly in `likes`/`dislikes` (no class subfolders), the model still trains preference.
- If you use class subfolders that match Fashion-MNIST names exactly, the classifier head will also be trained.
- Supported files: .jpg, .jpeg, .png, .bmp, .webp

## Recommended
- Keep images clear and centered on the clothing item.
- Prefer color images when using RGB (the notebook converts channels consistently).

## Optional: validate/prepare
Run the helper script to validate structure and produce a summary JSON:

```
python projects/clothing-preference-model/scripts/prepare_custom_data.py \
  --root projects/clothing-preference-model/data/custom \
  --summary projects/clothing-preference-model/data/custom/summary.json
```

Export processed copies (resize/channel convert) without touching originals:

```
python projects/clothing-preference-model/scripts/prepare_custom_data.py \
  --root projects/clothing-preference-model/data/custom \
  --export-dir projects/clothing-preference-model/data/custom_processed \
  --channels 3 --size 28
```

After adding images, open the notebook and run the training cells. It will detect and use your custom data automatically.
