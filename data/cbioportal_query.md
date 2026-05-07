# cBioPortal Data Query Instructions

Instructions for reproducing the ORIEN Avatar lung cancer dataset used in
figures 1a, 1b, and supplemental figure S1a.

---

## Study

**Study:** THO — Lung Cancer (ORIEN Avatar)
**URL:** https://www.cbioportal.org
**Expected cohort size:** 6,907 samples in 3,567 patients

---

## Step 1 — Query by gene

1. Navigate to cBioPortal and select the **THO — Lung Cancer** study
2. Click **Query by Gene**
3. Under **Select Genomic Profiles**, enable all of the following:
   - Mutations (WES)
   - Structural Variant
   - Discrete copy-number values
   - mRNA expression z-Scores (RNA Seq V2 RSEM)
4. Set the **z-score threshold** to `± 0`
5. Enter the following gene list:

```
HLA-DRA HLA-DRB1 HLA-DRB5 HLA-DPA1 HLA-DPB1 HLA-DQA1 HLA-DQA2 HLA-DQB1
HLA-DQB2 CIITA HLA-DOA HLA-DOB CD74 HLA-DMA HLA-DMB CD3D CD3E CD274 CD4
ROS1 EGFR ALK BRAF MET RET KRAS TP53
```

6. Click **Submit**

---

## Step 2 — Download expression data

1. Navigate to the **Download** tab
2. Select **mRNA expression (RNA Seq V2 RSEM)** — tab delimited format
3. Save as: `mRNA expression (RNA Seq V2 RSEM).txt`

> Note: if the file is too large to download in one piece, cBioPortal will
> split it. Download all parts and label them `mysig_pt1_...`, `mysig_pt2_...`
> etc. The loading code handles both single-file and multi-file cases.

---

## Step 3 — Download clinical and mutation data

Navigate to the **Plots** tab. For each comparison below, uncheck all boxes
under **Color samples by** at the top, then click the download button (cloud
icon) and select **Data**. Save each file with the name indicated.

| X axis | Y axis | Save as |
|--------|--------|---------|
| Clinical: OverallSurvivalStatus | Clinical: OverallSurvival (Months) | `overall_survival.txt` |
| Clinical: TumorMutationalBurden | Clinical: MicroSatelliteInstability | `tmb_microsatelite.txt` |
| Clinical: Cancer Type Details | Clinical: Stage | `cancer_type_vs_stage.txt` |
| Clinical: ImmunoOncologyDrug | Clinical: Sex | `immunotherapy_vs_sex.txt` |
| Clinical: AgeAtDiagnosis | Clinical: TumorMutationalBurden | `age_metas.txt` |
| Mutation: ROS1 (Mutated vs WT) | Mutation: EGFR (Mutated vs WT) | `ros1_egfr.txt` |
| Mutation: ALK (Mutated vs WT) | Mutation: BRAF (Mutated vs WT) | `alk_braf.txt` |
| Mutation: MET (Mutated vs WT) | Mutation: RET (Mutated vs WT) | `met_ret.txt` |
| Mutation: KRAS (Mutated vs WT) | Mutation: TP53 (Mutated vs WT) | `kras_tp53.txt` |

---

## Step 4 — Place files

Place all downloaded files in the directory specified by
`datasets.orien.root` in `data/paths_config.yml`. The expected
directory structure is:

```
{orien_root}/
├── data/
│   ├── mRNA expression (RNA Seq V2 RSEM).txt
│   ├── overall_survival.txt
│   ├── tmb_microsatelite.txt
│   ├── cancer_type_vs_stage.txt
│   ├── immunotherapy_vs_sex.txt
│   ├── age_metas.txt
│   ├── ros1_egfr.txt
│   ├── alk_braf.txt
│   ├── met_ret.txt
│   └── kras_tp53.txt
```

---

## Notes

- The expression matrix may be split into multiple part files by cBioPortal
  depending on cohort size at time of download. The loading code handles this
  automatically by globbing for `mysig_pt*` files or loading the single file.
- Column names in the clinical files may vary slightly between cBioPortal
  versions — check `data/paths_config.yml` and the loading cells if columns
  are not found.
- Cohort size may differ slightly from the values used in the paper if the
  ORIEN Avatar study has been updated since the original download.