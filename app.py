import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import warnings
import base64

# Suppress all warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

# --- SECURE ALPACA KEYS ---
API_KEY = st.secrets.get("ALPACA_API_KEY")
SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    st.error("❌ Clés API Alpaca manquantes. Veuillez configurer vos secrets Streamlit (ALPACA_API_KEY, ALPACA_SECRET_KEY).")
    st.stop()

# Initialize Alpaca client
@st.cache_resource
def get_alpaca_client():
    """
    Caches the Alpaca client to avoid re-initialization on every rerun.
    Uses st.cache_resource for objects that should be persisted across sessions.
    """
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

client = get_alpaca_client()

# --- Helper function for download button ---
def get_table_download_link(df, filename="data.csv", text_label="Télécharger les données en CSV"):
    """Generates a link to download a dataframe as a CSV file."""
    csv = df.to_csv(index=True)  # index=True to keep the timestamp as a column
    b64 = base64.b64encode(csv.encode()).decode()  # bytes <-> base64 <-> utf-8
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text_label}</a>'
    return href

# --- Streamlit Application UI ---
st.title("📈 Analyse de Données Boursières Historiques avec Alpaca")
st.markdown("""
    Cette application vous permet de récupérer et d'analyser les données historiques
    de plusieurs symboles boursiers via l'API Alpaca.
""")

# --- Configuration des symboles et de la période ---
st.sidebar.header("Paramètres de l'Analyse")

# Default symbols for demonstration
default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
symbols_input = st.sidebar.text_input(
    "Entrez les symboles boursiers (séparés par des virgules) :",
    value=", ".join(default_symbols)
)
selected_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
if not selected_symbols:
    st.sidebar.warning("Veuillez entrer au moins un symbole boursier.")
    st.stop()

# Date range selection
today = datetime.now().date()
default_start_date = today - timedelta(days=365)  # Last year
start_date = st.sidebar.date_input("Date de début :", value=default_start_date)
end_date = st.sidebar.date_input("Date de fin :", value=today)

if start_date >= end_date:
    st.sidebar.error("La date de début doit être antérieure à la date de fin.")
    st.stop()

# Timeframe selection
timeframe_options = {
    "1 Minute": TimeFrame.Minute,
    "15 Minutes": TimeFrame.Minute_15,
    "1 Heure": TimeFrame.Hour,
    "1 Jour": TimeFrame.Day,
}
selected_timeframe_str = st.sidebar.selectbox(
    "Sélectionnez la période :",
    options=list(timeframe_options.keys())
)
selected_timeframe = timeframe_options[selected_timeframe_str]

# --- Data Fetching Function ---
@st.cache_data(ttl=timedelta(hours=1))  # Cache data for 1 hour to reduce API calls
def fetch_stock_data(symbol, start, end, tf):
    """
    Fetches historical stock data for a given symbol and time range from Alpaca API.
    Handles potential API errors and returns a DataFrame.
    """
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end
        )
        bars = client.get_stock_bars(request_params).data
        if bars:
            df = pd.DataFrame([bar.dict() for bar in bars])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            if 'symbol' in df.columns:
                df.drop(columns=['symbol'], inplace=True)
            st.success(f"✔️ Données récupérées pour {symbol} avec succès.")
            return symbol, df
        else:
            st.warning(f"⚠️ Aucune donnée trouvée pour {symbol} dans la période spécifiée.")
            return symbol, pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération des données pour {symbol}: {e}")
        return symbol, pd.DataFrame()

# --- Main Data Processing and Display ---
if st.sidebar.button("Charger les données et Analyser"):
    st.subheader("🚀 Chargement des données...")
    all_stocks_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(fetch_stock_data, symbol, start_date, end_date, selected_timeframe): symbol
            for symbol in selected_symbols
        }
        fetched_count = 0
        for future in as_completed(future_to_symbol):
            symbol, df = future.result()
            if not df.empty:
                all_stocks_data[symbol] = df
            fetched_count += 1
            progress_bar.progress(fetched_count / len(selected_symbols))
            status_text.text(f"Récupération des données pour {symbol} ({fetched_count}/{len(selected_symbols)})...")

    progress_bar.empty()
    status_text.empty()

    if not all_stocks_data:
        st.warning("Aucune donnée n'a pu être chargée pour les symboles sélectionnés. Veuillez vérifier les symboles ou la période.")
        st.stop()

    st.success("✅ Toutes les données ont été chargées !")

    # Display Raw Data
    st.subheader("📊 Données Boursières Brutes")
    selected_display_symbol = st.selectbox(
        "Sélectionnez un symbole pour afficher ses données brutes :",
        options=list(all_stocks_data.keys()),
        key="raw_data_selector"  # Unique key for this selectbox
    )

    if selected_display_symbol and selected_display_symbol in all_stocks_data:
        df_to_display = all_stocks_data[selected_display_symbol]
        st.dataframe(df_to_display.head())
        st.line_chart(df_to_display['close'])

        # --- Download button for raw data ---
        st.download_button(
            label=f"Télécharger les données brutes de {selected_display_symbol} en CSV",
            data=df_to_display.to_csv(index=True).encode('utf-8'),
            file_name=f"{selected_display_symbol}_raw_data.csv",
            mime="text/csv",
            key=f"download_raw_{selected_display_symbol}"  # Unique key
        )

    st.subheader("📈 Analyse de Base et Visualisations")
    for symbol, df in all_stocks_data.items():
        st.markdown(f"---")
        st.markdown(f"### Analyse pour {symbol}")

        if not df.empty and 'close' in df.columns:
            st.write("#### Statistiques des prix de clôture :")
            st.write(df['close'].describe())

            df['daily_return'] = df['close'].pct_change() * 100
            st.write("#### Rendements Quotidiens (%)")
            st.line_chart(df['daily_return'].dropna())

            col_ret1, col_ret2 = st.columns(2)
            with col_ret1:
                st.info(f"Moyenne des rendements : `{df['daily_return'].mean():.2f}%`")
            with col_ret2:
                st.info(f"Volatilité (écart-type) : `{df['daily_return'].std():.2f}%`")

            window = st.slider(f"Sélectionnez la fenêtre pour la Moyenne Mobile Simple (SMA) pour {symbol}:",
                               min_value=5, max_value=50, value=20, key=f"sma_slider_{symbol}")
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
            st.write(f"#### Prix de Clôture et SMA {window}")
            st.line_chart(df[['close', f'SMA_{window}']].dropna())

            if 'volume' in df.columns:
                st.write("#### Volume des Transactions")
                st.bar_chart(df['volume'])
            else:
                st.warning(f"La colonne 'volume' n'est pas disponible pour {symbol}.")

            # --- Download button for analyzed individual stock data ---
            st.download_button(
                label=f"Télécharger l'analyse de {symbol} en CSV",
                data=df.to_csv(index=True).encode('utf-8'),
                file_name=f"{symbol}_analyzed_data.csv",
                mime="text/csv",
                key=f"download_analyzed_{symbol}"  # Unique key
            )
        else:
            st.warning(f"Impossible d'effectuer l'analyse pour {symbol} car les données sont manquantes ou incomplètes.")

    st.subheader("Comparable des Performances des Symboles")
    if len(all_stocks_data) > 1:
        normalized_data = pd.DataFrame()
        for symbol, df in all_stocks_data.items():
            if not df.empty and 'close' in df.columns and not df['close'].iloc[0] == 0:
                normalized_data[symbol] = (df['close'] / df['close'].iloc[0] - 1) * 100
            else:
                st.warning(f"Impossible de normaliser les données pour {symbol} (données manquantes ou prix initial zéro).")

        if not normalized_data.empty:
            st.line_chart(normalized_data)
            st.write("Ce graphique montre la performance normalisée en pourcentage pour chaque symbole, en commençant à 0%.")

            # --- Download button for normalized comparison data ---
            st.download_button(
                label="Télécharger les données de performance comparée en CSV",
                data=normalized_data.to_csv(index=True).encode('utf-8'),
                file_name="comparative_performance.csv",
                mime="text/csv",
                key="download_comparative_performance"  # Unique key
            )
        else:
            st.warning("Aucune donnée valide n'a pu être normalisée pour la comparaison. Vérifiez les symboles ou la période.")
    else:
        st.info("Ajoutez plus de symboles pour activer la comparaison des performances.")

    st.success("Analyse terminée ! 🎉 Explorez les données et les graphiques ci-dessus.")