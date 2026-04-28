# TransitRisk Data Generation Log

**START_DATE**: 2025-01-01  
**RANDOM_SEED**: 42  
**N_STATIONS**: 60  
**N_ROUTES**: 12  
**N_DAYS**: 90  
**TARGET_EVENTS**: 1247832  
**ACTUAL_EVENTS** (before DQ inject): 1247832  
**FINAL_ROWS** (after DQ inject): 1251575  
**N_STATION_ROUTE_PAIRS**: 176  

## Data Quality Injections
| Issue | Fraction | Approx Count |
|-------|----------|--------------|
| Exact duplicates | 0.3% | ~3744 |
| Negative delays | 0.15% | ~1872 |
| Future timestamps | 0.05% | ~624 |
| Precip NaN | 1.8% | ~22461 |
| Demand NaN | 1.2% | ~14974 |
| Outlier delays (120+) | 0.1% | ~1248 |
| Corrupt station_id | 0.5% | ~6239 |
| NaT timestamps | exact | 3 |