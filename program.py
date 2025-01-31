import streamlit as st
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, compare
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import statsmodels.stats.diagnostic as smd
from linearmodels.panel import PooledOLS
from statsmodels.stats.diagnostic import het_breuschpagan
import numpy as np
from scipy import stats
import csv
import os

# Funkcja do rysowania wykresów
def plot_corr_matrix(data):
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

def scatter_plot(data, x_var, y_var):
    sns.scatterplot(x=data[x_var], y=data[y_var])
    plt.title(f"Wykres zależności: {x_var} vs {y_var}")
    st.pyplot(plt)

def export_to_csv(data):
    if data.empty:
        st.warning("Brak danych do zapisania.")
        return None

    output = io.StringIO()
    data.to_csv(output, index=False, sep=";", encoding='utf-8')
    return output.getvalue()

# Ładowanie danych
@st.cache_data
def load_data():

    # data = pd.read_csv("Polpan1988_2018_v1.csv") 
    data = pd.read_csv("dane_pomniejszone.csv")

    # Filtrowanie danych
    required_columns = [
            "ANONID", "WAVE1988", "WAVE1993", "WAVE1998", "WAVE2003", "WAVE2008", "WAVE2013", "WAVE2018",
            "Z103", "YR13", "XB24", "WR1621", "VB19", "UG29", "TG18",
            "Z119", "YR40", "XR42", "WR42", "VR33", "UK36", "TK33",
            "JOB1988_CUR", "JOB1993_CUR", "JOB1998_CUR", "JOB2003_CUR", "JOB2008_CUR", "JOB2013_CUR", "JOB2018_CUR",
            "EDUC1988", "EDUC1993", "EDUC1998", "EDUC2003", "EDUC2008", "EDUC2013", "EDUC2018", "Z02_PSCO",
            "YA081_PSCO", "XB01_PSCO", "WB01_PSCO", "VB05PSCO", "UF01_SKZ", "TJ00_PSCO",
            "WW01", "AGE1988"

    ]
       
    # Sprawdź, czy wszystkie wymagane kolumny istnieją
    if not set(required_columns).issubset(data.columns):
        st.error("Brakuje wymaganych kolumn w danych.")
        return pd.DataFrame()  # Zwróć pusty DataFrame

    # Filtrowanie wierszy bez brakujących wartości w wymaganych kolumnach
    filtered_data = data[data[required_columns].notnull().all(axis=1)]

    return filtered_data

# Ładowanie danych do session_state
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

# Menu nawigacyjne
st.sidebar.title("Nawigacja")
menu = st.sidebar.radio("Wybierz sekcję:", [
    "Wprowadzenie",
    "Prezentacja i opis danych",
    "Budowa, weryfikacja i podsumwoanie modelu"
])

# Globalne zmienne
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'X' not in st.session_state:
    st.session_state['X'] = None

if menu == "Wprowadzenie":
    st.title("Projekt zaliczeniowy: Ekonometria danych panelowych")
    st.write("Prowadząca: Justyna Tora")
    st.write("Autorzy: Natalia Łyś, Zuzanna Deszcz")
    st.header("Wprowadzenie")
    # Wprowadzenie w formacie Markdown
    st.markdown(
        """
        ## Polskie Badanie Panelowe (POLPAN)

        **POLPAN** to unikatowy projekt badawczy realizowany przez Instytut Filozofii i Socjologii Polskiej Akademii Nauk. Od **1988 roku** zbiera dane na temat struktury społecznej w Polsce, dokumentując kluczowe zmiany społeczne i gospodarcze.

        ### Zakres badania obejmuje:
        - **Historię zawodową** respondentów
        - **Opinie i postawy** społeczne, gospodarcze i polityczne
        - **Relacje rodzinne** i sieci społeczne
        - **Zdrowie fizyczne i psychiczne**
        - **Korzystanie z technologii informacyjnych**

        Dzięki **regularnym pomiarom co 5 lat**, POLPAN umożliwia śledzenie indywidualnych trajektorii życiowych i analizowanie długoterminowych trendów w społeczeństwie polskim.
   
        W ramach projektu POLPAN analizowaliśmy różne czynniki, które mają wpływ na poziom dochodów w Polsce. Badanie pozwala na identyfikację kluczowych determinant ekonomicznych, takich jak wykształcenie, doświadczenie zawodowe, płeć, pochodzenie społeczne czy dostęp do rynków pracy.
        Praca z danymi była bardzo długa i żmudna. Przejrzałyśmy wszystkie kwestionariusze na przestrzeni lat, która nie są storzone jednolicie i dopasowałyśmy odpowiedzi z akrusza do zmiennych w bazie danych.
        [Oficjalna strona POLPAN](https://polpan.org/en/)
        """,
        unsafe_allow_html=True
    )


elif menu == "Prezentacja i opis danych":
    st.header("Prezentacja i opis danych")

    st.write("Dane pozyskaniu interesujących nas kolumn ze zbioru danych zostały pobrane i to z takiego pomniejszonego zbioru danych korzystamy. Github nie pozwolił na przesłanie tak dużego pliku jak dane z POLPAN. Interesowali nas tylko i wyłącznie, Ci uczestnicy, którzy brali udział we wszystkich edycjach POLPAN. Daje to około 550 osób. Kolumny z brakami zostały usunięte.")
    if st.session_state['data'] is not None and not st.session_state['data'].empty:
        data = st.session_state['data']
        st.write("Podgląd danych:")
     
        remaining_columns = [
            "ANONID", "WAVE1988", "WAVE1993", "WAVE1998", "WAVE2003", "WAVE2008", "WAVE2013", "WAVE2018",
            "Z103", "YR13", "XB24", "WR1621", "VB19", "UG29", "TG18",
            "Z119", "YR40", "XR42", "WR42", "VR33", "UK36", "TK33",
            "JOB1988_CUR", "JOB1993_CUR", "JOB1998_CUR", "JOB2003_CUR", "JOB2008_CUR", "JOB2013_CUR", "JOB2018_CUR",
            "EDUC1988", "EDUC1993", "EDUC1998", "EDUC2003", "EDUC2008", "EDUC2013", "EDUC2018", "Z02_PSCO",
            "YA081_PSCO", "XB01_PSCO", "WB01_PSCO", "VB05PSCO", "UF01_SKZ", "TJ00_PSCO",
            "WW01", "AGE1988"
 
        ]
        
        missing_columns = set(remaining_columns) - set(data.columns)
        if missing_columns:
            st.error(f"Brakuje następujących wymaganych kolumn: {missing_columns}")
            st.stop()
            
        mapa_wartosci_uk36 = {
        1: 10,
        2: 30,
        3: 50,
        4: 100,
        5: 300,
        6: 500,
        7: 1000,
        88: 0
        }

        # Zastosowanie mapowania i uzupełnienie wartości nieznanych (NaN) zerem
        data['UK36'] = data['UK36'].map(mapa_wartosci_uk36).fillna(0)  
        
        mapa_wartosci_tk33 = {
        1: 10,
        2: 30,
        3: 50,
        4: 100,
        5: 300,
        6: 500,
        7: 1000,
        88: 0,
        -11:0
        }

        # Zastosowanie mapowania i uzupełnienie wartości nieznanych (NaN) zerem
        data['TK33'] = data['TK33'].map(mapa_wartosci_tk33).fillna(0)  
        

        data_cleaned = data[remaining_columns].copy()
        data_cleaned.dropna(inplace=True)
        

        # Lista lat, dla których chcemy stworzyć kolumny z wiekiem
        years = [1993, 1998, 2003, 2008, 2013, 2018]
        for year in years:
            data_cleaned[f'AGE{year}'] = data_cleaned['AGE1988'] + (year - 1988)

        # Sprawdź wynik
        st.write("Dane z nowymi kolumnami wieku:")
        st.dataframe(data_cleaned.head())

        # 1. Zdefiniuj mapowanie kolumn -> poszczególne lata
        mappings = {
            1988: {
                "wave_col": "WAVE1988",
                "income_col": "Z103",
                "job_col": "JOB1988_CUR",
                "educ_col": "EDUC1988",
                "prof_col": "Z02_PSCO",
                "books_owned":  "Z119", 
                "gender": "WW01",
                "age": "AGE1988",
            },
            1993: {
                "wave_col": "WAVE1993",
                "income_col": "YR13",
                "job_col": "JOB1993_CUR",
                "educ_col": "EDUC1993",
                "prof_col": "YA081_PSCO",
                "books_owned":  "YR40",
                "gender": "WW01",
                "age": "AGE1993",
            },
            1998: {
                "wave_col": "WAVE1998",
                "income_col": "XB24",
                "job_col": "JOB1998_CUR",
                "educ_col": "EDUC1998",
                "prof_col": "XB01_PSCO",
                "books_owned": "XR42",
                "gender": "WW01",
                "age": "AGE1998",
            },
            2003: {
                "wave_col": "WAVE2003",
                "income_col": "WR1621",
                "job_col": "JOB2003_CUR",
                "educ_col": "EDUC2003",
                "prof_col": "WB01_PSCO",
                "books_owned": "WR42", 
                "gender": "WW01",
                "age": "AGE2003",
            },
            2008: {
                "wave_col": "WAVE2008",
                "income_col": "VB19",
                "job_col": "JOB2008_CUR",
                "educ_col": "EDUC2008",
                "prof_col": "VB05PSCO",
                "books_owned": "VR33",
                "gender": "WW01",
                "age": "AGE2008",
            },
            2013: {
                "wave_col": "WAVE2013",
                "income_col": "UG29",
                "job_col": "JOB2013_CUR",
                "educ_col": "EDUC2013",
                "prof_col": "UF01_SKZ",
                "books_owned": "UK36",
                "gender": "WW01",
                "age": "AGE2013",
            },
            2018: {
                "wave_col": "WAVE2018",
                "income_col": "TG18",
                "job_col": "JOB2018_CUR",
                "educ_col": "EDUC2018",
                "prof_col": "TJ00_PSCO",
                "books_owned": "TK33",
                "gender": "WW01",
                "age": "AGE2018",
            },
        }

        #  konwertuje dane "szerokie" do formatu panelowego
        def convert_to_panel(df_wide, mapping_dict):
            df_list = []

            for year, cols in mapping_dict.items():
                # Wybierz potrzebne kolumny
                sub_df = df_wide[[
                    "ANONID",
                    cols["wave_col"],
                    cols["income_col"],
                    cols["job_col"],
                    cols["educ_col"],
                    cols["prof_col"],
                    cols["books_owned"],
                    cols["gender"],  
                    cols["age"]
                ]].copy()

                # Zmień nazwy, aby je ujednolicić
                sub_df.rename(columns={
                    cols["wave_col"]: "wave_indicator",
                    cols["income_col"]: "income",
                    cols["job_col"]: "job_status",
                    cols["educ_col"]: "education",
                    cols["prof_col"]: "profession",
                    cols["books_owned"]: "owned_books",
                    cols["gender"] : "gender",
                    cols["age"] : "age"
                }, inplace=True)

                # Dodaj kolumnę z rokiem
                sub_df["wave"] = year

                df_list.append(sub_df)

            # Połącz wszystkie 'sub_df' w jeden DataFrame
            panel_data = pd.concat(df_list, ignore_index=True)
            # Posortuj wg ANONID i wave
            panel_data.sort_values(by=["ANONID", "wave"], inplace=True)

            return panel_data

        # Zamiana na długie (panel)
        data_cleaned = convert_to_panel(data_cleaned, mappings)
        panel_df = data_cleaned.copy()
        
        # Interakcje
        panel_df['education_gender_interaction'] = panel_df['education'] * panel_df['gender']
        panel_df['proffesion_gender_interaction'] = panel_df['profession'] * panel_df['gender']
        panel_df['age_education_interaction'] = panel_df['age'] * panel_df['education']  

        ### Inflacja 

        # Dane o inflacji (wskaźniki cen przy podstawie rok poprzedni = 100)
        inflacja = {
            1988: 160.2, 1993: 135.3, 1998: 111.8, 2003: 100.8, 2008: 104.2, 2013: 100.9, 2018: 101.6
        }

        # Przekształcenie danych o inflacji w DataFrame
        inflacja_df = pd.DataFrame(list(inflacja.items()), columns=['Rok', 'Inflacja'])

        # Obliczenie indeksu cenowego (CPI) z bazowym rokiem 2018
        inflacja_df['CPI'] = 1.0  
        for i in range(1, len(inflacja_df)):
            inflacja_df.loc[i, 'CPI'] = inflacja_df.loc[i - 1, 'CPI'] * (inflacja_df.loc[i, 'Inflacja'] / 100)

        # Normalizacja CPI do roku bazowego 2018
        bazowy_rok = 2018
        cpi_bazowy = inflacja_df.loc[inflacja_df['Rok'] == bazowy_rok, 'CPI'].values[0]
        inflacja_df['CPI'] = inflacja_df['CPI'] / cpi_bazowy

        # Połączenie danych o inflacji z danymi panelowymi
        panel_df = pd.merge(panel_df, inflacja_df, left_on='wave', right_on='Rok', how='left')

        # Deflacja dochodu
        panel_df['income_deflated'] = panel_df['income'] / panel_df['CPI']

        
        st.markdown(
        """
        ### Normalizacja wskaźnika CPI i deflacja dochodu

        Aby umożliwić porównywanie dochodów w różnych latach, wskaźnik **CPI (Consumer Price Index)** został znormalizowany względem roku bazowego **2018**. Wartości **CPI** podzielono przez jego poziom w roku 2018, dzięki czemu CPI w roku bazowym przyjmuje wartość **1**, a dla pozostałych lat odzwierciedla względne zmiany cen. Następnie dane inflacyjne połączono z danymi panelowymi na podstawie roku badania. Dochód nominalny został przeliczony na wartości realne poprzez podzielenie go przez znormalizowany wskaźnik **CPI**, zgodnie z wzorem:
        """)

        st.latex(r"""
        Dochód^{real}_t = \frac{Dochód^{nom}_t}{CPI_{t}^{norm}}
        """)

        st.markdown("""
        Dzięki temu wszystkie dochody wyrażone są w cenach z **2018 roku**, co pozwala na ich porównywanie w czasie oraz uwzględnienie efektów inflacyjnych w analizach ekonomicznych.

        [Roczne wskaźniki cen towarów i usług konsumpcyjnych od 1950 roku](https://stat.gov.pl/obszary-tematyczne/ceny-handel/wskazniki-cen/wskazniki-cen-towarow-i-uslug-konsumpcyjnych-pot-inflacja-/roczne-wskazniki-cen-towarow-i-uslug-konsumpcyjnych/)
        """,
        unsafe_allow_html=True
        )
        
        # Sprawdź wynik
        st.write("Dane z deflowanym dochodem:")
        
        st.dataframe(panel_df[['wave', 'income', 'CPI', 'income_deflated']].head())
        
        # Dane PKB per capita dla Polski
        dane_pkbpercapita = {
            "Rok": [1988, 1993, 1998, 2003, 2008, 2013, 2018],
            "PKB_per_capita": [12809.89, 12551.76, 17540.15, 20859.80, 28495.91, 34323.58, 43585.12]
        }

       # Tworzenie DataFrame
        pkb_df = pd.DataFrame(dane_pkbpercapita)

        # Przeliczenie na ceny stałe z 2018 w złotych
        kurs_ppp_2021 = 1.92  # 1$ PPP = 1.92 PLN w 2021
        deflator_2018_2021 = 1.10  # Wskaźnik inflacji między 2018 a 2021

        pkb_df["PKB_per_capita_2018_PLN"] = (pkb_df["PKB_per_capita"] * kurs_ppp_2021) / deflator_2018_2021

        # Wyświetlenie tabeli w Streamlit

        st.markdown("""
        ### PKB per capita w cenach stałych z 2018 roku

        Dane przedstawiają **PKB per capita** dla Polski w latach **1988–2018** w dolarach międzynarodowych, które zostały przeliczone na **złote w cenach stałych z 2018 roku**.
        [Polska - PKB per capita](https://pl.tradingeconomics.com/poland/gdp-per-capita)

        """, unsafe_allow_html=True) 


        st.dataframe(pkb_df)
        
        # Dane PKB per capita dla Polski
        pkb_data = {
            "wave": [1988, 1993, 1998, 2003, 2008, 2013, 2018],  
            "PKB_per_capita": [12809.89, 12551.76, 17540.15, 20859.80, 28495.91, 34323.58, 43585.12]
        }

        # Tworzenie DataFrame z danymi o PKB
        pkb_df = pd.DataFrame(pkb_data)

        # Połączenie danych o PKB z danymi panelowymi
        panel_df = pd.merge(panel_df, pkb_df, on='wave', how='left')
        
        
        
        st.subheader("Dane panelowe:")
        st.write("Dane przekształcone do formatu danych panelych. Wiersze z brakującymi wartościami zostały usunięte.")
        st.dataframe(panel_df)

        st.write(f"Wierszy: {panel_df.shape[0]}, Kolumn: {panel_df.shape[1]}")

        # Przetwarzanie dalszych danych
        panel_df = panel_df.drop(columns=["wave_indicator", "job_status"])
        panel_df = panel_df[panel_df["income"] <= 9000]
       

        # Sprawdzenie zmienności zmiennych
        st.subheader("Sprawdzenie zmienności zmiennych")
        st.write("Sprawdzamy, które zmienne są czasowo zmienne dla poszczególnych jednostek.")

        # Funkcja sprawdzająca zmienność zmiennych
        def check_variability(df, group_col, var_cols):
            variability = {}
            for var in var_cols:
                unique_counts = df.groupby(group_col)[var].nunique()
                variability[var] = unique_counts.gt(1).any()
            return variability

        var_to_check = ["education", "profession"]
        variability = check_variability(panel_df, "ANONID", var_to_check)

        for var, is_variable in variability.items():
            if not is_variable:
                st.warning(f"Zmienna `{var}` jest czasowo niezmienna i zostanie usunięta z modeli Fixed Effects.")
            else:
                st.success(f"Zmienna `{var}` jest czasowo zmienna.")

        # Usunięcie zmiennych czasowo niezmiennych
        vars_to_remove = [var for var, is_var in variability.items() if not is_var]
        if vars_to_remove:
            panel_df = panel_df.drop(columns=vars_to_remove)
            st.write(f"Usunięto zmienne czasowo niezmienne: {vars_to_remove}")

        # Store panel_df w session_state dla innych sekcji
        st.session_state['panel_df'] = panel_df

        st.header("Statystyki opisowe")
        panel_df["profession"] = panel_df["profession"].astype("category")
        panel_df["gender"] = panel_df["gender"].astype("category")

        # Logarytmowanie dochodu
        panel_df['log_income_deflated'] = np.log(panel_df['income_deflated'] + 1) 
    

        # Wyświetlamy statystyki opisowe dla wybranych kolumn numerycznych
        numeric_cols = ["income", "education", "owned_books", "age","PKB_per_capita"] 
        st.write(panel_df[numeric_cols].describe())
        
        st.header("Histogram wybranej zmiennej")

        # Wybór zmiennej do histogramu
        var_for_hist = st.selectbox("Wybierz zmienną do analizy rozkładu:",
                                    options=numeric_cols + ["profession"],
                                    index=0)

        # Rysujemy histogram (Plotly Express)
        if var_for_hist != "profession":
            fig_hist = px.histogram(
                panel_df,
                x=var_for_hist,
                nbins=30,  
                title=f"Histogram zmiennej {var_for_hist}"
            )
        else:
            fig_hist = px.histogram(
                panel_df,
                x=var_for_hist,
                color=var_for_hist,
                title=f"Histogram zmiennej {var_for_hist}"
            )
        st.plotly_chart(fig_hist)

 
        st.header("Trend w czasie / Fale badania")

    
        mean_income_by_wave = (
            panel_df.groupby("wave", as_index=False)["income_deflated"]
            .mean()
            .rename(columns={"income_deflated": "mean_income_deflated"})
        )

        fig_income_wave = px.line(
            mean_income_by_wave,
            x="wave",
            y="mean_income_deflated",
            markers=True,
            title="Średni dochód w zależności od fali (wave)"
        )
        st.plotly_chart(fig_income_wave)
        
        st.write("Wykres przedstawia cykliczność lub zmienność dochodów w czasie, co może być wynikiem różnych czynników, takich jak polityka gospodarcza, cykle koniunkturalne, inflacja (mimo deflowania dochodu) lub zmiany strukturalne w gospodarce.Szczególną uwagę zwraca dynamiczny wzrost do 1995 roku i późniejszy gwałtowny spadek, co mogłoby wymagać głębszej analizy przyczyn tych zmian.")
        
        # Wykres zależności dochodu od wieku
        fig = px.scatter(panel_df, x='age', y='log_income_deflated', color='gender', title="Dochód w zależności od wieku i płci")
        st.plotly_chart(fig)

        st.write("Wykres wskazuje na klasyczny wzorzec dochodów rosnących wraz z wiekiem, aż do osiągnięcia szczytu w wieku średnim, a następnie spadku w starszym wieku. Występują różnice dochodów między płciami, szczególnie w wieku produkcyjnym. Może to być przedmiotem dalszych analiz dotyczących nierówności płacowych, wpływu urlopów związanych z opieką nad dziećmi lub innych czynników społeczno-ekonomicznych.")

        st.write("Podstawowe statystyki:")
        st.write(panel_df.describe())

        st.subheader("Macierz korelacji:")
        df_num = panel_df[numeric_cols].copy()
        plot_corr_matrix(df_num)
        
        st.write("Dochód jest najsilniej skorelowany z wykształceniem, co potwierdza, że edukacja ma znaczący wpływ na zarobki. Wiek jest silnie powiązany z poziomem PKB na mieszkańca, co może sugerować, że w bardziej rozwiniętych regionach populacja jest starsza. Liczba posiadanych książek wydaje się nie mieć istotnego wpływu na inne zmienne, co może być związane z różnymi preferencjami kulturowymi lub poziomem dochodów.")

        fig = px.box(
            panel_df,
            x="profession",      
            y="log_income_deflated",      
            title="Dochód w zależności od zawodu (profession)"
        )
        st.plotly_chart(fig)
        st.write("Dochody różnią się istotnie w zależności od grup zawodowych. Grupy o kodzie zbliżonym do 5000 wydają się mieć największy potencjał do wysokich dochodów. Rozrzut w niektórych zawodach sugeruje dużą różnorodność w wynagrodzeniach, co może być związane z różnymi poziomami stanowisk w danej grupie zawodowej.")
        st.session_state['data_cleaned'] = data_cleaned

elif menu == "Budowa, weryfikacja i podsumwoanie modelu":
    st.header("Budowa modelu")
    panel_df = st.session_state['panel_df']

    panel_df = panel_df.set_index(["ANONID", "wave"])
    
    # Sprawdzenie zmienności zmiennych niezależnych
    def check_variability(df, group_col, var_cols):
        variability = {}
        for var in var_cols:
            unique_counts = df.groupby(group_col)[var].nunique()
            variability[var] = unique_counts.gt(1).any()
        return variability
    
    # standaryzacja
    scaler = StandardScaler()
    panel_df[['education', 'age', 'owned_books', 'PKB_per_capita', 'log_income_deflated']] = scaler.fit_transform(panel_df[['education', 'age', 'owned_books', 'PKB_per_capita', 'log_income_deflated']])
    
    # Nowy zestaw zmiennych
    X_vars = ["education", "owned_books", "gender", "age", "education_gender_interaction", "proffesion_gender_interaction", "age_education_interaction", "profession", "PKB_per_capita"]
    variability = check_variability(panel_df, "ANONID", X_vars)
    vars_to_remove = [var for var, is_var in variability.items() if not is_var]
    panel_df_backup = panel_df.copy()
    if vars_to_remove:
        panel_df = panel_df.drop(columns=vars_to_remove)
        X_vars = [var for var in X_vars if var not in vars_to_remove]
        st.warning(f"Usunięto zmienne czasowo niezmienne: {vars_to_remove}")
    
    # Definiowanie zmiennej zależnej i niezależnych
    y = panel_df["log_income_deflated"]
    X = panel_df[X_vars]
    X = add_constant(X)
    
    # Pooled OLS
    pooled_model = PooledOLS(y, X).fit(cov_type='clustered', cluster_entity=True)
    
    # Random Effects (Efekty losowe)
    re_model = RandomEffects(y, X).fit(cov_type='clustered', cluster_entity=True)
    
    # Fixed Effects (Efekty ustalone - jednostkowe) z drop_absorbed=True
    fe_model = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    
    # Fixed Effects z efektami czasowymi (dwukierunkowy)
    fe_tw_model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    
    # Wyświetlenie wyników modeli
    st.subheader("Porównanie modeli panelowych")
    model_comparison = compare({"Pooled OLS": pooled_model, "Random Effects": re_model, 
                                "Fixed Effects": fe_model, "Fixed Effects (2-way)": fe_tw_model})
    st.write(model_comparison)
    
    # Test Breusch-Pagana (czy Random Effects > Pooled OLS)
    bp_test_stat = (re_model.resids.var() - pooled_model.resids.var()) / pooled_model.resids.var() * (len(y) - len(X.columns))
    bp_p_value = 1 - stats.chi2.cdf(bp_test_stat, df=1)
    st.subheader("Test Breusch-Pagana")
    st.write(f"Wartość statystyki: {bp_test_stat:.4f}, p-wartość: {bp_p_value:.4f}")
    
    # Test Hausmana (czy Fixed Effects > Random Effects)
    hausman_stat = np.sum((fe_model.params - re_model.params)**2 / (fe_model.std_errors**2 + re_model.std_errors**2))
    hausman_p_value = 1 - stats.chi2.cdf(hausman_stat, df=len(X.columns))
    st.subheader("Test Hausmana")
    st.write(f"Wartość statystyki: {hausman_stat:.4f}, p-wartość: {hausman_p_value:.4f}")
    
    # Test efektów jednostkowych
    fe_f_test = fe_model.f_statistic
    st.subheader("Test efektów indywidualnych")
    st.write(f"Wartość statystyki F: {fe_f_test.stat:.4f}, p-wartość: {fe_f_test.pval:.4f}")
    
    # Test efektów czasowych
    fe_tw_f_test = fe_tw_model.f_statistic
    st.subheader("Test efektów czasowych")
    st.write(f"Wartość statystyki F: {fe_tw_f_test.stat:.4f}, p-wartość: {fe_tw_f_test.pval:.4f}")
    
    # Wybór najlepszego modelu
    st.subheader("Wybór najlepszego modelu")
    if bp_p_value < 0.05:
        if hausman_p_value < 0.05:
            if fe_tw_f_test.pval < 0.05:
                st.success("Model Fixed Effects z efektami czasowymi jest najlepszy.")
            else:
                st.success("Model Fixed Effects jest najlepszy.")
        else:
            st.success("Model Random Effects jest najlepszy.")
    else:
        st.success("Model Pooled OLS jest najlepszy.")

    
    st.title("Analiza wyników modeli panelowych")

    st.markdown("""
    ## Wprowadzenie
    Model jest raczej mierny w przewidywaniu dochodów. Zmienne, które zostały pobrane ze zbioru danych nie mają na tyle istotnego wpływu.
    Mimo tego, że najlpesza jest regresja łącz

    """)

    best_model = pooled_model  # Pooled OLS jako najlepszy model

    # Reszty modelu
    residuals = best_model.resids

    st.title("Analiza modelu ekonometrycznego")

    st.subheader("Testy diagnostyczne")

    # Test na autokorelację składnika losowego (test Wooldridge'a)
    def wooldridge_autocorrelation_test(model, panel_df):
        df = panel_df.copy()
        df['residuals'] = model.resids
        df['residuals_lagged'] = df.groupby(level=0)['residuals'].shift(1)
        df.dropna(inplace=True)
        
        result = sm.OLS(df['residuals'], sm.add_constant(df['residuals_lagged'])).fit()
        test_stat = result.tvalues[1]
        p_value = result.pvalues[1]
        return test_stat, p_value

    test_stat, p_value = wooldridge_autocorrelation_test(best_model, panel_df)
    st.write(f"Test Wooldridge'a na autokorelację: statystyka = {test_stat:.4f}, p-wartość = {p_value:.4f}")

    # Test na heteroskedastyczność (test Breuscha-Pagana)
    residuals_sq = residuals ** 2
    X_resid = sm.add_constant(X)
    bp_test = smd.het_breuschpagan(residuals_sq, X_resid)
    st.write(f"Test Breuscha-Pagana na heteroskedastyczność: Chi2 = {bp_test[0]:.4f}, p-wartość = {bp_test[1]:.4f}")

    # Test normalności reszt (Jarque-Bera)
    jb_test = jarque_bera(residuals)
    st.write(f"Test Jarque-Bera na normalność reszt: Chi2 = {jb_test[0]:.4f}, p-wartość = {jb_test[1]:.4f}")

    st.subheader("Wykresy diagnostyczne")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram reszt
    sns.histplot(residuals, bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Histogram reszt modelu")

    # Wykres normalności (Q-Q plot)
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title("Q-Q plot reszt")

    # Wykres reszt vs. wartości dopasowane
    axes[2].scatter(best_model.fitted_values, residuals, alpha=0.5)
    axes[2].axhline(y=0, color='red', linestyle='--')
    axes[2].set_xlabel("Wartości dopasowane")
    axes[2].set_ylabel("Reszty")
    axes[2].set_title("Reszty vs. wartości dopasowane")

    plt.tight_layout()
    st.pyplot(fig)

    if p_value < 0.05 and bp_test[1] < 0.05:
        st.warning("Autokorelacja i heteroskedastyczność wykryte")
    elif p_value < 0.05:
        st.warning("Autokorelacja wykryta.")
    elif bp_test[1] < 0.05:
        st.warning("Heteroskedastyczność wykryta")
        
    # Analiza wyników
    st.markdown("""
    ### Analiza wyników modelu ekonometrycznego

    #### **Testy diagnostyczne**
    - Test Wooldridge'a **{:.4f}** (p-wartość: {:.4f}), wykazuje {} autokorelację.
    - Test Breuscha-Pagana **Chi2 = {:.4f}** (p-wartość: {:.4f}), wykazuje {} heteroskedastyczność.
    - Test Jarque-Bera: **Chi2 = {:.4f}**, p-wartość: {:.4f}, wskazuje {}, że reszty mają rozkład normalny.


    """.format(test_stat, p_value, "wykryto" if p_value < 0.05 else "nie wykryto", 
                bp_test[0], bp_test[1], "wykryto" if bp_test[1] < 0.05 else "nie wykryto", 
                jb_test[0], jb_test[1], "sugeruje" if jb_test[1] > 0.05 else "nie sugeruje", 
                "poprawne" if jb_test[1] > 0.05 else "niepoprawne"))

