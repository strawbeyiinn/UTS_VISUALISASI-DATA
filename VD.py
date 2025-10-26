import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ["order_date", "registered_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = ["price","qty_ordered","before_discount","discount_amount","after_discount","cogs"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    bd = df["before_discount"] if "before_discount" in df.columns else (df.get("price", 0) * df.get("qty_ordered", 0))
    after_disc_col = df.get("after_discount", pd.Series([0]*len(df)))
    disc_amount_col = df.get("discount_amount", pd.Series([0]*len(df)))
    price_col = df.get("price", pd.Series([0]*len(df)))
    qty_col = df.get("qty_ordered", pd.Series([0]*len(df)))
    cogs_col = df.get("cogs", pd.Series([0]*len(df)))
    df["value_sales"] = np.where(bd>0, bd, price_col * qty_col)
    df["revenue"]     = np.where(after_disc_col>0, after_disc_col, df["value_sales"] - disc_amount_col)
    df["profit"]      = (price_col - cogs_col) * qty_col - disc_amount_col
    df["margin_pct"]  = np.where(df["revenue"]>0, df["profit"]/df["revenue"]*100, np.nan)
    for c in ["category","payment_method","campaign","sku_name","brand"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def idr(x):
    try:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return x

def add_month(df, date_col="order_date"):
    d = df.copy()
    if date_col in d.columns:
        d = d.dropna(subset=[date_col])
        d["month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        d["month"] = pd.NaT
    return d

def auto_trend_comment(monthly_df, value_col="revenue"):
    md = monthly_df.dropna(subset=["month"])
    if md.empty:
        return "Data 2022 tidak tersedia untuk membentuk tren."
    md = md.sort_values("month").reset_index(drop=True)
    md["t"] = np.arange(len(md))
    slope_rev = np.polyfit(md["t"], md[value_col].fillna(0), 1)[0] if len(md) >= 2 else 0
    slope_ord = np.polyfit(md["t"], md["orders"].fillna(0), 1)[0] if "orders" in md.columns and len(md) >= 2 else 0
    bullets = []
    if slope_rev > 0:
        bullets.append("• **Trend revenue 2022 cenderung naik** (momentum positif).")
    elif slope_rev < 0:
        bullets.append("• **Trend revenue 2022 cenderung turun** (perlu tindakan korektif).")
    else:
        bullets.append("• **Trend revenue 2022 relatif datar**.")
    if slope_ord > 0:
        bullets.append("• **Orders meningkat** sepanjang 2022.")
    elif slope_ord < 0:
        bullets.append("• **Orders menurun** sepanjang 2022.")
    else:
        bullets.append("• **Orders relatif stabil** sepanjang 2022.")
    aov0 = (md.loc[0, value_col] / md.loc[0, "orders"]) if md.loc[0, "orders"]>0 else np.nan
    aovN = (md.loc[len(md)-1, value_col] / md.loc[len(md)-1, "orders"]) if md.loc[len(md)-1, "orders"]>0 else np.nan
    if pd.notna(aov0) and pd.notna(aovN):
        if aovN > aov0:
            bullets.append("• **AOV akhir > awal** (monetisasi per order membaik).")
        elif aovN < aov0:
            bullets.append("• **AOV akhir < awal** (perlu optimasi monetisasi).")
    cta = []
    if slope_rev < 0 or slope_ord < 0:
        cta.append("1) Audit campaign dengan share besar tapi **AOV rendah** → revisi offer/targeting.")
        cta.append("2) Aktifkan **re-engagement** (remarketing) untuk segmen customer dorman.")
        cta.append("3) Uji **promo terarah** (min. spend, bundling) untuk mengangkat AOV & conversion.")
    else:
        cta.append("1) **Scale** campaign dengan ROI positif; tingkatkan budget bertahap (10–20%).")
        cta.append("2) Lakukan **A/B test** pada kreatif/landing untuk menambah conversion tanpa diskon besar.")
        cta.append("3) Eksplor **cross-selling** & bundling demi mengangkat AOV.")
    note = "  \n".join(bullets) + "\n\n**Call to Action (CTA):**\n" + "\n".join([f"- {x}" for x in cta])
    return note

st.set_page_config(page_title="VISDAT UTS Dashboard", layout="wide")
st.title("📊 VISDAT UTS Dashboard")

with st.sidebar:
    st.header("⚙️ Data & Filter")
    data_source = st.text_input("CSV path", "DATASET_UTS.csv")
    df = load_data(data_source)
    if "order_date" in df.columns and df["order_date"].notna().any():
        years = sorted(df["order_date"].dt.year.dropna().unique().tolist())
        year_selected = st.multiselect("Tahun (opsional)", years, default=years)
        df = df[df["order_date"].dt.year.isin(year_selected)]
    else:
        st.warning("Kolom order_date tidak ditemukan atau kosong.")
    if "category" in df.columns:
        cats = sorted(df["category"].dropna().unique().tolist())
        cat_selected = st.multiselect("Kategori (opsional)", cats, default=cats[:1] if cats else [])
        if cat_selected:
            df = df[df["category"].isin(cat_selected)]
    campaign_field = "campaign" if "campaign" in df.columns else ("payment_method" if "payment_method" in df.columns else None)

tab1, tab2 = st.tabs(["📢 Campaign Performance", "📊 Products Sales"])

with tab1:
    st.subheader("📢 Campaign Performance Dashboard")
    if "order_date" in df.columns:
        d22 = df[df["order_date"].dt.year == 2022].copy()
    else:
        d22 = df.copy()
    total_orders = d22["id"].nunique() if "id" in d22.columns else len(d22)
    total_customers = d22["customer_id"].nunique() if "customer_id" in d22.columns else np.nan
    value_sales = d22["value_sales"].sum()
    revenue = d22["revenue"].sum()
    profit = d22["profit"].sum()
    aov = value_sales / total_orders if total_orders else np.nan
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Orders (2022)", f"{total_orders:,}")
    col2.metric("Customers (2022)", f"{total_customers:,}" if pd.notna(total_customers) else "—")
    col3.metric("Value Sales (2022)", f"Rp {idr(value_sales)}")
    col4.metric("Revenue (2022)", f"Rp {idr(revenue)}")
    col5.metric("AOV (2022)", f"Rp {idr(aov)}" if pd.notna(aov) else "—")
    monthly = (
        add_month(d22)
        .groupby("month", as_index=False)
        .agg(orders=("id","nunique"),
             value_sales=("value_sales","sum"),
             revenue=("revenue","sum"))
        .sort_values("month")
    )
    monthly["AOV"] = np.where(monthly["orders"]>0, monthly["value_sales"]/monthly["orders"], np.nan)
    st.markdown("**Tren Bulanan 2022 (Orders, Revenue, AOV)**")
    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["orders"], name="Orders")
    fig.add_bar(x=monthly["month"], y=monthly["revenue"], name="Revenue")
    fig.add_scatter(x=monthly["month"], y=monthly["AOV"], mode="lines+markers", name="AOV", yaxis="y2")
    fig.update_layout(barmode="group", xaxis_title="Month", yaxis=dict(title="Orders / Revenue"), yaxis2=dict(title="AOV", overlaying="y", side="right"), legend=dict(orientation="h", y=1.2), margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)
    name_field = "sku_name" if "sku_name" in d22.columns else ("product_name" if "product_name" in d22.columns else None)
    if name_field:
        group_cols = [name_field]
        if "category" in d22.columns:
            group_cols.append("category")
        prod = (
            d22.groupby(group_cols, as_index=False)
               .agg(before_discount=("value_sales","sum"),
                    after_discount=("revenue","sum"),
                    net_profit=("profit","sum"),
                    quantity=("qty_ordered","sum") if "qty_ordered" in d22.columns else ("id","count"),
                    unique_customers=("customer_id","nunique") if "customer_id" in d22.columns else ("id","nunique"),
                    orders=("id","nunique"))
               .sort_values("after_discount", ascending=False)
               .reset_index(drop=True)
        )
        prod["AOV"] = np.where(prod["orders"]>0, prod["after_discount"] / prod["orders"], np.nan)
        st.markdown("**Tabel Performa Produk (2022)**")
        show = prod.copy()
        for c in ["before_discount","after_discount","net_profit","AOV"]:
            show[c] = show[c].apply(idr)
        st.dataframe(show, use_container_width=True)
        st.markdown("**Top 10 Produk berdasarkan Revenue (2022)**")
        top10 = prod.nlargest(10, "after_discount")
        bar = px.bar(top10, x="after_discount", y=name_field, orientation="h", text="after_discount", title="Top 10 Products — After Discount Revenue (2022)")
        bar.update_traces(texttemplate="Rp %{text:,.0f}", textposition="outside", cliponaxis=False)
        bar.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(bar, use_container_width=True)
        st.markdown("**Tren Bulanan Produk Terpilih (2022)**")
        chosen = st.selectbox("Pilih produk", top10[name_field] if len(top10)>0 else prod[name_field])
        df_sel = d22[d22[name_field]==chosen]
        msel = (
            add_month(df_sel)
            .groupby("month", as_index=False)
            .agg(orders=("id","nunique"),
                 revenue=("revenue","sum"),
                 profit=("profit","sum"))
            .sort_values("month")
        )
        line = go.Figure()
        line.add_scatter(x=msel["month"], y=msel["revenue"], mode="lines+markers", name="Revenue")
        line.add_scatter(x=msel["month"], y=msel["profit"], mode="lines+markers", name="Profit")
        line.update_layout(xaxis_title="Month", yaxis_title="Amount", legend=dict(orientation="h", y=1.2))
        st.plotly_chart(line, use_container_width=True)
    st.markdown("### 📌 Ringkasan Tren & Rekomendasi Aksi (CTA)")
    st.info(auto_trend_comment(monthly, value_col="revenue"))

with tab2:
    st.subheader("📊 Product Sales Dashboard")
    if "order_date" in df.columns:
        d22 = df[df["order_date"].dt.year == 2022].copy()
    else:
        d22 = df.copy()

    subset = d22.copy()
    qty = subset["qty_ordered"].sum() if "qty_ordered" in subset.columns else len(subset)
    unique_cust = subset["customer_id"].nunique() if "customer_id" in subset.columns else np.nan
    orders = subset["id"].nunique() if "id" in subset.columns else len(subset)

    c1, c2, c3 = st.columns(3)
    c1.metric("Quantity (2022, All Categories)", f"{int(qty):,}")
    c2.metric("Unique Customers", f"{unique_cust:,}" if pd.notna(unique_cust) else "—")
    c3.metric("Orders", f"{orders:,}")

    if not subset.empty:
        cols_show = [c for c in ["order_date","category","payment_method","campaign","sku_name","qty_ordered","revenue","profit","customer_id","id"] if c in subset.columns]
        st.markdown("**Detail Transaksi (2022, All Categories)**")
        detail = subset[cols_show].copy()
        for c in ["revenue","profit"]:
            if c in detail.columns:
                detail[c] = detail[c].apply(idr)
        st.dataframe(detail, use_container_width=True, height=320)
    else:
        st.warning("Tidak ditemukan transaksi pada tahun 2022.")

    st.divider()
    st.markdown("### Share Orders per Payment (2022)")
    if campaign_field:
        by_cmp_22 = (
            d22.groupby(campaign_field, dropna=False)
               .agg(orders=("id","nunique"))
               .sort_values("orders", ascending=False)
               .reset_index()
        )
        left, right = st.columns([1,1])
        with left:
            st.dataframe(by_cmp_22, use_container_width=True)
        with right:
            st.plotly_chart(px.pie(by_cmp_22, values="orders", names=campaign_field, hole=0.5), use_container_width=True)

