# SFR2PAR

**SFR2PAR** is a Fortran utility that parameterizes streambed conductivities in a MODFLOW 2000/2005/NWT Streamflow-Routing Package (SFR) file using multipliers for each stream segment. Optionally, it can look up the base vertical hydraulic conductivity (Kv) from a MODFLOW LPF/UPW file and multiply it by user-supplied segment-specific factors.

It reads a previously created SFR file, updates the streambed conductivities for each reach, and then copies the remainder of the previous SFR file into the new one.

This is particularly useful for parameter estimation workflows (e.g., PEST), where you want to systematically adjust streambed conductance values.

## Execution & Usage

### Main Input File
The SFR2PAR control file is expected to have:
```
<Original SFR file name>
<New (updated) SFR file name>
<LPF/UPW file name or "NONE">
nlay  nrow  ncol
segID1   multiplier1
segID2   multiplier2
...
segIDN   multiplierN
```
- A multiplier value of -999 is used to indicate to SFR2PAR that the value currently present in the SFR file should be used (no multiplier, etc)
- The code expects you to provide nseg lines, each containing an integer segment ID and a real multiplier value.
**Important**: The segment ID in the control file must match the SFR file’s ISEG (or segment IDs) for the multiplier to be applied correctly.

### Running SFR2PAR
```bash
SFR2PAR <sfr2par_input_file>
```
- SFR2PAR reads the <sfr2par_input_file> and processes the SFR and LPF/UPW files accordingly.
- Upon completion, it writes the updated SFR file named as specified in line 2 of the control file.

## Limitations & Assumptions
### SFR Assumptions
- **ICALC = 1**: SFR2PAR assumes a standard SFR data layout consistent with ICALC=1, meaning:
  - SFR2PAR reads a fixed set of data fields per reach, specifically: `KRCH IRCH JRCH ISEG IREACH RCHLEN STRTOP SLOPE STRTHICK STRHC1`
  - No separate flow routing equations or time-varying stage/discharge equations are considered here.
- **Options/Headers:** The program skips commented lines (#) and certain lines starting with T or R. It also expects any OPTIONS block to appear before Data Set 1c.
- **Segment IDs:** We assume the SFR file’s segment IDs (ISEG) match the integer IDs in the SFR2PAR control file. Any mismatch could result in incorrect indexing or missing multipliers.
### LPF/UPW Assumptions
- **Fixed Array Format**: SFR2PAR assumes the arrays present in the LPF/UPW file are `HYDRAULIC CONDUCTIVITY`, `VERTICAL HYD. COND.`, `PRIMARY STORAGE`, and `SPECIFIC YIELD` (for each layer)
- **No Parameter Blocks:** The code does not handle LPF/UPW “PARAMETER” definitions. All arrays are assumed to be read from standard “INTERNAL” lines.

## Example
See the [test](./test/) folder for an example.
