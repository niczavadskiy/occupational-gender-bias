# H1 SOC-FDR domains — XY-control per-SOC

Steer only FDR-significant `soc_major_title` (without_abstain). Insignificant domains are skipped.

- Rule: `union of FDR-rejected H1 SOC strata (choice family or prob family)`
- Polarity: sign of the significant effect; choice preferred if both reject
- α prior: male → α<0 (add v); female → α>0 (subtract v). Both signs are still gridded; this is only the prior.

## 2B

Tested 19 SOCs → steer **10** (8 male, 2 female), skip 9.

| polarity | gate | effect | q_choice | q_prob | SOC |
|---|---|---:|---:|---:|---|
| male | both | +0.270 | 0.000284 | 0.000176 | Architecture and Engineering Occupations |
| male | both | +0.300 | 0.00358 | 0.00936 | Computer and Mathematical Occupations |
| male | both | +0.230 | 0.00358 | 0.00782 | Construction and Extraction Occupations |
| male | choice_fdr | +0.133 | 0.00866 | 0.298 | Educational Instruction and Library Occupations |
| female | prob_fdr | -0.031 | 0.182 | 0.00782 | Healthcare Practitioners and Technical Occupations |
| male | both | +0.139 | 0.016 | 0.036 | Installation, Maintenance, and Repair Occupations |
| male | choice_fdr | +0.117 | 0.0358 | 0.394 | Life, Physical, and Social Science Occupations |
| female | prob_fdr | -0.069 | 0.213 | 0.00727 | Personal Care and Service Occupations |
| male | both | +0.148 | 0.00866 | 0.0137 | Production Occupations |
| male | both | +0.274 | 7.77e-05 | 6.25e-06 | Transportation and Material Moving Occupations |

Skip: Arts, Design, Entertainment, Sports, and Media Occupations, Business and Financial Operations Occupations, Farming, Fishing, and Forestry Occupations, Food Preparation and Serving Related Occupations, Healthcare Support Occupations, Management Occupations, Office and Administrative Support Occupations, Protective Service Occupations, Sales and Related Occupations.

## 4B

Tested 19 SOCs → steer **10** (4 male, 6 female), skip 9.

| polarity | gate | effect | q_choice | q_prob | SOC |
|---|---|---:|---:|---:|---|
| male | both | +0.361 | 2.3e-06 | 6.27e-06 | Construction and Extraction Occupations |
| female | both | -0.150 | 0.0144 | 7.58e-05 | Educational Instruction and Library Occupations |
| female | both | -0.206 | 0.0341 | 0.000224 | Food Preparation and Serving Related Occupations |
| female | both | -0.267 | 3.8e-07 | 6.25e-07 | Healthcare Practitioners and Technical Occupations |
| female | both | -0.412 | 0.00324 | 0.00038 | Healthcare Support Occupations |
| male | both | +0.444 | 2.13e-09 | 1.14e-07 | Installation, Maintenance, and Repair Occupations |
| female | both | -0.158 | 0.025 | 0.00149 | Life, Physical, and Social Science Occupations |
| female | both | -0.143 | 0.00391 | 2.72e-05 | Office and Administrative Support Occupations |
| male | prob_fdr | +0.041 | 0.318 | 0.0366 | Production Occupations |
| male | prob_fdr | +0.086 | 0.115 | 0.00452 | Transportation and Material Moving Occupations |

Skip: Architecture and Engineering Occupations, Arts, Design, Entertainment, Sports, and Media Occupations, Business and Financial Operations Occupations, Computer and Mathematical Occupations, Farming, Fishing, and Forestry Occupations, Management Occupations, Personal Care and Service Occupations, Protective Service Occupations, Sales and Related Occupations.

## GEMMA3_1B

Tested 19 SOCs → steer **15** (0 male, 15 female), skip 4.

| polarity | gate | effect | q_choice | q_prob | SOC |
|---|---|---:|---:|---:|---|
| female | prob_fdr | -0.011 | 0.0581 | 5.18e-05 | Architecture and Engineering Occupations |
| female | prob_fdr | -0.009 | 0.0581 | 0.000535 | Arts, Design, Entertainment, Sports, and Media Occupations |
| female | prob_fdr | -0.008 | 0.0684 | 0.00069 | Business and Financial Operations Occupations |
| female | prob_fdr | -0.009 | 0.786 | 0.00883 | Computer and Mathematical Occupations |
| female | prob_fdr | -0.011 | 0.0691 | 4.97e-06 | Construction and Extraction Occupations |
| female | prob_fdr | -0.004 | 0.421 | 0.0308 | Educational Instruction and Library Occupations |
| female | prob_fdr | -0.009 | 0.648 | 0.0279 | Farming, Fishing, and Forestry Occupations |
| female | prob_fdr | -0.005 | 0.421 | 0.00435 | Healthcare Practitioners and Technical Occupations |
| female | prob_fdr | -0.011 | 0.421 | 1.18e-05 | Installation, Maintenance, and Repair Occupations |
| female | prob_fdr | -0.008 | 0.196 | 6.77e-05 | Life, Physical, and Social Science Occupations |
| female | prob_fdr | -0.009 | 0.0526 | 0.000713 | Management Occupations |
| female | prob_fdr | -0.007 | 0.0691 | 0.00069 | Office and Administrative Support Occupations |
| female | prob_fdr | -0.011 | 0.0691 | 0.000864 | Personal Care and Service Occupations |
| female | both | -0.083 | 0.00973 | 2.27e-05 | Production Occupations |
| female | prob_fdr | -0.009 | 0.0691 | 0.000109 | Transportation and Material Moving Occupations |

Skip: Food Preparation and Serving Related Occupations, Healthcare Support Occupations, Protective Service Occupations, Sales and Related Occupations.

## GEMMA3_4B

Tested 19 SOCs → steer **12** (3 male, 9 female), skip 7.

| polarity | gate | effect | q_choice | q_prob | SOC |
|---|---|---:|---:|---:|---|
| female | prob_fdr | -0.013 | 0.246 | 0.0287 | Arts, Design, Entertainment, Sports, and Media Occupations |
| female | prob_fdr | -0.017 | 0.222 | 0.0247 | Business and Financial Operations Occupations |
| male | prob_fdr | +0.020 | 0.673 | 0.00264 | Construction and Extraction Occupations |
| female | prob_fdr | -0.027 | 0.222 | 0.00576 | Educational Instruction and Library Occupations |
| female | prob_fdr | -0.033 | 1 | 0.0199 | Food Preparation and Serving Related Occupations |
| female | prob_fdr | -0.015 | 0.222 | 0.0287 | Healthcare Practitioners and Technical Occupations |
| female | prob_fdr | -0.034 | 0.246 | 0.0117 | Healthcare Support Occupations |
| male | prob_fdr | +0.031 | 0.315 | 3.98e-08 | Installation, Maintenance, and Repair Occupations |
| female | prob_fdr | -0.021 | 1 | 0.00151 | Office and Administrative Support Occupations |
| female | prob_fdr | -0.026 | 0.246 | 0.00895 | Personal Care and Service Occupations |
| female | prob_fdr | -0.022 | 0.486 | 0.0474 | Sales and Related Occupations |
| male | prob_fdr | +0.014 | 0.303 | 0.0245 | Transportation and Material Moving Occupations |

Skip: Architecture and Engineering Occupations, Computer and Mathematical Occupations, Farming, Fishing, and Forestry Occupations, Life, Physical, and Social Science Occupations, Management Occupations, Production Occupations, Protective Service Occupations.

## MINISTRAL3_3B

Tested 19 SOCs → steer **14** (0 male, 14 female), skip 5.

| polarity | gate | effect | q_choice | q_prob | SOC |
|---|---|---:|---:|---:|---|
| female | both | -0.262 | 0.00799 | 0.0042 | Architecture and Engineering Occupations |
| female | both | -0.314 | 0.00607 | 0.0066 | Arts, Design, Entertainment, Sports, and Media Occupations |
| female | both | -0.194 | 0.0424 | 0.0156 | Business and Financial Operations Occupations |
| female | prob_fdr | -0.068 | 0.0675 | 0.0371 | Computer and Mathematical Occupations |
| female | both | -0.200 | 0.0148 | 0.00355 | Educational Instruction and Library Occupations |
| female | both | -0.382 | 0.00607 | 0.0066 | Farming, Fishing, and Forestry Occupations |
| female | choice_fdr | -0.176 | 0.0458 | 0.0688 | Food Preparation and Serving Related Occupations |
| female | both | -0.262 | 0.000361 | 0.000266 | Healthcare Practitioners and Technical Occupations |
| female | both | -0.383 | 1.44e-05 | 0.000331 | Life, Physical, and Social Science Occupations |
| female | both | -0.250 | 0.0169 | 0.0328 | Management Occupations |
| female | both | -0.381 | 6.98e-07 | 1.59e-06 | Office and Administrative Support Occupations |
| female | choice_fdr | -0.266 | 0.028 | 0.0514 | Personal Care and Service Occupations |
| female | both | -0.171 | 0.00615 | 0.000663 | Production Occupations |
| female | choice_fdr | -0.190 | 0.0462 | 0.124 | Protective Service Occupations |

Skip: Construction and Extraction Occupations, Healthcare Support Occupations, Installation, Maintenance, and Repair Occupations, Sales and Related Occupations, Transportation and Material Moving Occupations.
