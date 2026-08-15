## Data Augmentation

### Quran Augmentation

Synthetic Quran examples were generated from the **quranlab/quran** dataset.

Generation includes:

- Random introductory phrases (e.g., *قال تعالى*)
- Single or multiple consecutive verses
- Different quotation styles
- Multiple claimed source templates
- Arabic and Eastern Arabic numerals
- Automatic span annotation for:
  - `ayah`
  - `claimed_source`

### Hadith Augmentation

Synthetic Hadith examples were generated from the **Sanadset** dataset.

Generation includes:

- Extraction of `SANAD` and `MATN`
- Arabic text normalization
- Automatically generated claimed-source statements using multiple templates
- Automatic span annotation for:
  - `isnad`
  - `matn`
  - `claimed_source`

### Hard Negative Examples

Twenty hard negative passages were manually selected from classical **Shamela** books.

These passages were intentionally chosen because they contain words such as **"قال"** and resemble Hadith-style narration, while **not** containing actual Hadith spans nor Ayah. They are included to help models distinguish ordinary Arabic narrative text from authentic Hadith structures and reduce false positive predictions.

## Data Augmentation Code

The scripts used to generate the augmented data are publicly available:

- [Quran augmentation script](task1_span_detection/scripts/data_augmentation/quran_data_augmentation.py)

- [Hadith augmentation script](task1_span_detection/scripts/data_augmentation/hadith_data_augmentation.py)

- [Hard negative selection tool](task1_span_detection/scripts/data_augmentation/hard_negative_selector_app.py)

These scripts reproduce the synthetic Quran, Hadith and Hard Negative examples and their corresponding Task 1 span annotations.

## Source Datasets
- Official IslamicEval 2026 Task 1 training set
- Sanadset: https://huggingface.co/datasets/arbml/Sanadset
- Quran: https://huggingface.co/datasets/quranlab/quran
- Shamela books: https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB
