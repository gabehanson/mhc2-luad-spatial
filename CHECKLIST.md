# Pre-submission checklist

Steps to complete before making this repository public.
Work through these in order after all analyses are finalized.

---

## 1. Complete all analyses

- [ ] Figure 2 scrna notebooks — done
- [ ] Figure 2e — done
- [ ] mhc2-patient-classification — done
- [ ] CosMx spatial transcriptomics notebooks
- [ ] Remaining figure notebooks (figures 3, 4, 5...)
- [ ] Migrate notebook-local helper functions into `ceiba.plot_utils`

---

## 2. Clean and finalize code

- [ ] Remove all hardcoded absolute paths — everything should resolve via `paths_config.yml`
- [ ] Remove all exploratory/scratch cells from notebooks
- [ ] Confirm all notebooks have markdown headers matching the reference style
- [ ] Confirm `ceiba` package imports correctly from a fresh environment

---

## 3. Run all notebooks end to end

- [ ] Restart kernel and run all cells top to bottom for each notebook
- [ ] Confirm no errors on clean run
- [ ] Confirm all output files are saved to `outputs/figures/` and `outputs/tables/`

---

## 4. Execute notebooks with inline outputs for GitHub rendering

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/scrna/*.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/cosmx/*.ipynb
```

- [ ] Verify figures render correctly in GitHub notebook preview
- [ ] Check repo size is acceptable after adding outputs

---

## 5. Check for sensitive data in outputs

- [ ] No patient IDs or clinical metadata in notebook outputs
- [ ] No institutional file paths in printed output cells
- [ ] No credentials or API keys anywhere in the repo

---

## 6. Update environment and dependencies

- [ ] Export final conda environment: `conda env export > environment.yml`
- [ ] Pin dependency versions in `setup.py`
- [ ] Test install from scratch: `pip install -e .`

---

## 7. Write README

- [ ] Project description and paper citation
- [ ] Installation instructions
- [ ] Data availability statement
- [ ] Instructions for updating `paths_config.yml` for new machines
- [ ] Notebook descriptions and figure mapping

---

## 8. Final repo hygiene

- [ ] Tag a release matching paper submission: `git tag v1.0.0-submission`
- [ ] Confirm `.gitignore` is catching all large data files
- [ ] Confirm no raw data files were accidentally committed: `git ls-files | grep -E '\.h5ad|\.csv|\.h5'`

---

## 9. Make repository public

- [ ] After acceptance or at preprint submission
- [ ] Update README with DOI and citation once available