# Lexicons

The Loughran-McDonald Master Dictionary (1993-2025) is **not** stored in this repository: its licence allows academic use and
asks that commercial users obtain a licence, so the file is kept with the owner's inputs at
`DATA_ROOT/inputs/lexicons/Loughran-McDonald_MasterDictionary_1993-2025.csv`.

- Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (Google Sheet export; URL in `configs/models.yaml`)
- Pinned checksum (sha256, verified 2026-10-23): `147cb24a1687ccde60363a0fa29c62be4097debcc698c01a83e51bb8a176dbb3`
- If the file is missing, S09 downloads it from the export URL and refuses to continue when the checksum differs.
