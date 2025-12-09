import pandas as pd

def summarize_results(results):
    df = pd.DataFrame(results)
    print("\n===== RESULTS SUMMARY =====")
    print(df)
    print("===========================\n")

    # If it's a CV result
    if "RMSE_mean" in df.columns:
        best = df.loc[df["RMSE_mean"].idxmin()]
        print(f"Best model: {best['model']} (RMSE_mean={best['RMSE_mean']:.3f} ± {best['RMSE_std']:.3f})")
    else:
        best = df.loc[df["RMSE"].idxmin()]
        print(f"Best model: {best['model']} (RMSE={best['RMSE']:.3f})")
