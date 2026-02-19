import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import re

# ---------- helpers ----------
def normalize_place(s: str) -> str:
    """
    Normalize city names:
    - lowercase
    - remove accents
    - strip state abbreviation in parentheses e.g. 'Abadia de Goiás (GO)' -> 'abadia de goias'
    - remove extra spaces/punctuation
    """
    if pd.isna(s):
        return np.nan
    s = str(s).strip()

    # *** KEY FIX: remove anything in parentheses like " (GO)" ***
    s = re.sub(r"\s*\(.*?\)", "", s)

    s = s.lower()

    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # keep letters/numbers/spaces only
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- 1) load datasets ----------
df  = pd.read_csv("2.1_City_Daily_Missing.csv")
soc = pd.read_excel("data.xlsx")

# ---------- 2) pick socioeconomic variables ----------
soc = soc.rename(columns={"Territorialidades": "City"})

income_col = "IDHM Renda 2010"
edu_col    = "IDHM Educação 2010"

soc = soc[["City", income_col, edu_col]].copy()

# ---------- 3) normalize city keys ----------
df["city_key"]  = df["City"].apply(normalize_place)
soc["city_key"] = soc["City"].apply(normalize_place)

# DEBUG: show sample keys to confirm they now align
print("Medical city_key examples:  ", df["city_key"].dropna().unique()[:10].tolist())
print("Soc     city_key examples:  ", soc["city_key"].dropna().unique()[:10].tolist())

merged = df.merge(
    soc.drop(columns=["City"]),
    on="city_key",
    how="left"
)

match_pct = merged[income_col].notna().mean() * 100
print(f"\nRows: {len(merged)}")
print(f"Soc matched (%): {match_pct:.1f}%")

if match_pct < 10:
    print("\n⚠️  Match rate still low. Showing unmatched medical city keys for inspection:")
    unmatched = merged.loc[merged[income_col].isna(), "city_key"].unique()
    print(unmatched[:30])

# ---------- 4) city-month aggregation ----------
merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
merged["month"] = merged["Date"].dt.to_period("M").astype(str)

city_month = (
    merged
    .groupby(["city_key", "month"], as_index=False)
    .agg(
        City         = ("City",             "first"),
        State        = ("State",            "first"),
        avg_ct       = ("Ct_Value",         "mean"),
        avg_stringency = ("Stringency_Index", "mean"),
        n_patients   = ("Patience_Count",   "sum"),
        income       = (income_col,         "first"),
        education    = (edu_col,            "first"),
    )
)

city_month = city_month[city_month["n_patients"] >= 10].copy()

# ---------- 5) income quartiles ----------
# *** KEY FIX: drop NaN income rows before qcut and add duplicates='drop' as safety ***
city_month = city_month.dropna(subset=["income", "avg_ct", "avg_stringency"]).copy()

if len(city_month) == 0:
    raise RuntimeError(
        "No rows remain after dropping NaN income. "
        "The city join failed completely — check city name format differences above."
    )

city_month["income_quartile"] = pd.qcut(
    city_month["income"],
    q=4,
    labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    duplicates="drop"   # safety valve in case of ties at bin edges
)

# ---------- 6) scatter plot ----------
colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]
fig, ax = plt.subplots(figsize=(10, 7))

for (q, sub), color in zip(city_month.groupby("income_quartile"), colors):
    ax.scatter(
        sub["avg_stringency"],
        sub["avg_ct"],
        label=str(q),
        alpha=0.65,
        s=40,
        color=color,
        edgecolors="white",
        linewidths=0.4
    )

ax.set_xlabel("Stringency Index (city-month average)", fontsize=12)
ax.set_ylabel("Average Ct Value (city-month)", fontsize=12)
ax.set_title(
    "Does Stringency Reduce Viral Load Equally Across Socioeconomic Groups?\n"
    "(Each dot = one city in one month)",
    fontsize=13
)
ax.legend(title="Income quartile (IDHM Renda 2020)", framealpha=0.9)
plt.tight_layout()
plt.savefig("stringency_vs_ct_by_income.png", dpi=150)
plt.show()
print("Plot saved to stringency_vs_ct_by_income.png")

# ---------- 7) interaction model ----------
try:
    import statsmodels.formula.api as smf

    city_month["stringency_c"] = city_month["avg_stringency"] - city_month["avg_stringency"].mean()
    city_month["income_c"]     = city_month["income"]         - city_month["income"].mean()
    city_month["education_c"]  = city_month["education"]      - city_month["education"].mean()

    model = smf.ols(
        "avg_ct ~ stringency_c + income_c + education_c + stringency_c:income_c + stringency_c:education_c",
        data=city_month
    ).fit()
    print(model.summary())

    # Interpretation guide
    print("\n--- Interpretation ---")
    coef_interaction = model.params.get("stringency_c:income_c", None)
    if coef_interaction is not None:
        if coef_interaction > 0:
            print("Positive interaction: stringency is LESS effective at reducing viral load in higher-income cities.")
        else:
            print("Negative interaction: stringency is MORE effective at reducing viral load in higher-income cities.")

except ImportError:
    print("Install statsmodels: pip install statsmodels")