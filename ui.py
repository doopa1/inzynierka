import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

# odczytanie plików

fill_defaults = pd.read_csv("fill_defaults.csv", index_col=0)
labels_for_ui = pd.read_csv("labels_for_ui.csv")
labels = pd.read_csv("df_usable.csv")

#__________________________________________________________________________________________________________________
# Ładuję model
#__________________________________________________________________________________________________________________

model = joblib.load("rf_model.pkl")




#__________________________________________________________________________________________________________________
# laduje dataset i koordynaty
#__________________________________________________________________________________________________________________
labels = pd.read_csv("df_usable.csv")
coords = pd.read_csv("city_coordinates_ranges.csv")


# Create a nicely styled visible Streamlit button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        padding: 0.75em 1.5em;
        border-radius: 5px;
        font-weight: bold;
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)

predict_clicked = st.button("Oblicz cenę")




#___________________________________________________

price_placeholder = st.markdown("""
                                position: fixed;
        top: 100px;
        right: 30px;""")




#rysuje linie
def line():
    st.markdown("<hr style='border:1px solid black'>", unsafe_allow_html=True)





# Function to get internal column name by UI label
def get_internal_col(ui_label):
    return labels_for_ui.loc[labels_for_ui['ui_label'] == ui_label, 'column'].values[0]


# Get internal column names using UI labels
city_col = get_internal_col("Miasto")
ownership_col = get_internal_col("Forma własności")

boolean_map = {"Posiada": 1, "Nie posiada": 0}


# typ wlasnosci - mapowanie
ownership_display_map = {
    "condominium": "Kondominium (współwłasność)",
    "cooperative": "Spółdzielnia Mieszkaniowa",
    "udział": "Udział"
}

# odwrocenie mapy
ownership_reverse_map = {v: k for k, v in ownership_display_map.items()}

# Get available internal ownership values from the dataset
internal_col_for_ownership = labels_for_ui.loc[labels_for_ui['ui_label'] == 'Forma własności', 'column'].values[0]
available_ownership_values = sorted(labels[internal_col_for_ownership].dropna().unique())







# mapowanie miast
city_col = labels_for_ui.loc[labels_for_ui["ui_label"] == "Miasto", "column"].values[0]

# dostepne miasta
available_cities = sorted(labels[city_col].dropna().unique())










#_________________________________________________________________________________________
#
#USER INPUT
#
#___________________________________________________________________________________________


st.title("Predykcja Ceny Mieszkania")

#________________________________________________________________________________
# inputy zawsze aktywne
#_______________________________________________________________________________



# Read city order from coordinates CSV
ordered_cities = coords["Miasto"].tolist()

# Get cities from main dataset
available_cities = labels[city_col].dropna().unique()

# Filter ordered cities to only those in available_cities
filtered_ordered_cities = [city for city in ordered_cities if city in available_cities]

# miasto
city = st.selectbox("Miasto", options=filtered_ordered_cities)
#city = st.selectbox("Miasto", options=sorted(labels[city_col].dropna().unique()), key="city_selectbox")

line()

#powierzchnia
square_meters = st.number_input("Powierzchnia (m²)", min_value=1.0, step=0.1, format="%.1f", value=30.0)

# inputy z checkboxem

line()
enable_rooms = st.checkbox("Liczba pokoi (włącz)", value=False)
if enable_rooms:
    rooms = st.number_input("Liczba pokoi", min_value=1, max_value=10, step=1)
else:
    st.write("Liczba pokoi — nieaktywne")
    rooms = None

line()
enable_floor_count = st.checkbox("Liczba pięter w budynku (włącz)", value=False)
if enable_floor_count:
    floor_count = st.number_input("Liczba pięter w budynku", min_value=1, max_value=52, step=1)
else:
    st.write("Liczba pięter w budynku — nieaktywne")
    floor_count = None

line()
enable_floor = st.checkbox("Piętro (włącz)", value=False)
if enable_floor and floor_count is not None:
    floor = st.number_input("Piętro", min_value=0, max_value=floor_count, step=1)
else:
    st.write("Piętro — nieaktywne")
    floor = None

# Ownership display with checkbox
line()
enable_ownership = st.checkbox("Forma własności (włącz)", value=False)
ownership_display_options = [ownership_display_map.get(val, val) for val in available_ownership_values]

if enable_ownership:
    selected_ownership_display = st.selectbox("Forma własności", options=ownership_display_options)
    ownership = ownership_reverse_map.get(selected_ownership_display, selected_ownership_display)
else:
    st.write("Forma własności — nieaktywne")
    ownership = None

# Binary choices with checkboxes

line()
enable_has_parking = st.checkbox("Miejsce parkingowe (włącz)", value=False)
if enable_has_parking:
    has_parking = st.selectbox("Miejsce parkingowe", options=boolean_map.keys())
else:
    st.write("Miejsce parkingowe — nieaktywne")
    has_parking = None

line()
enable_has_balcony = st.checkbox("Balkon (włącz)", value=False)
if enable_has_balcony:
    has_balcony = st.selectbox("Balkon", options=boolean_map.keys())
else:
    st.write("Balkon — nieaktywne")
    has_balcony = None

line()
enable_has_elevator = st.checkbox("Winda (włącz)", value=False)
if enable_has_elevator:
    has_elevator = st.selectbox("Winda", options=boolean_map.keys())
else:
    st.write("Winda — nieaktywna")
    has_elevator = None

line()
enable_has_security = st.checkbox("Ochrona (włącz)", value=False)
if enable_has_security:
    has_security = st.selectbox("Ochrona", options=boolean_map.keys())
else:
    st.write("Ochrona — nieaktywna")
    has_security = None

line()
enable_has_storage = st.checkbox("Komórka lokatorska (włącz)", value=False)
if enable_has_storage:
    has_storage = st.selectbox("Komórka lokatorska", options=boolean_map.keys())
else:
    st.write("Komórka lokatorska — nieaktywna")
    has_storage = None

# Numeric fields with checkboxes

line()
enable_build_year = st.checkbox("Rok budowy (włącz)", value=False)
if enable_build_year:
    build_year = st.number_input("Rok budowy", min_value=1800, max_value=2024, step=1)
else:
    st.write("Rok budowy — nieaktywny")
    build_year = None




#____________________________________________________________
# SZEROKOSC I DLUGOSC GEO
#____________________________________________________________


# pobranie wartosci min i max dl i szer geo
city_range = coords[coords["Miasto"] == city].iloc[0]
min_lat = city_range["Minimalna Szerokość"]
max_lat = city_range["Maksymalna Szerokość"]
min_lon = city_range["Minimalna Długość"]
max_lon = city_range["Maksymalna Długość"]


def geo_input(label, min_val, max_val, key_prefix):
    enable = st.checkbox(f"{label} (włącz)", value=False, key=f"{key_prefix}_enable")
    if enable:
        val = st.number_input(label, min_value=min_val, max_value=max_val, format="%.2f", key=f"{key_prefix}_input")
    else:
        st.write(f"{label} — nieaktywny/-a")
        val = None
    return val

line()
latitude = geo_input("Szerokość geograficzna", min_lat, max_lat, "latitude")
line()
longitude = geo_input("Długość geograficzna", min_lon, max_lon, "longitude")
# Distances with checkboxes

#____________________________________________________________________
# DYSTANSE
#______________________________________________________________________


def distance_input(label, key):
    container = st.container()
    with container:
        line()
        enable = st.checkbox(f"Aktywuj {label}", value=False, key=f"{key}_enable")
        if enable:
            val = st.number_input(label + " (m)", min_value=0.0, key=f"{key}_input") * 0.001
        else:
            st.write(f"{label} — nieaktywny/-a")
            val = None
        
        return val

# --- Now define all labels in a list ---
distance_labels = [
    "Odległość od centrum",
    "Odległość od szkoły",
    "Odległość od przychodni",
    "Odległość od poczty",
    "Odległość od przedszkola",
    "Odległość od restauracji",
    "Odległość od uczelni",
    "Odległość od apteki"
]

# --- Loop through labels, display inputs with container ---
distances = {}
for i, label in enumerate(distance_labels):
    key = f"distance_{i}"
    distances[label] = distance_input(label, key=key)
 
#distances = {}
#for i, label in enumerate(distance_labels):
#    distances[label] = distance_input(label, key=f"distance_checkbox_{i}")
#    line()

# POI count with checkbox
line()
enable_poi_count = st.checkbox("Liczba ważnych punktów w okolicy (włącz)", value=False)
if enable_poi_count:
    poi_count = st.number_input("Liczba ważnych punktów w okolicy", min_value=0, step=1)
else:
    st.write("Liczba ważnych punktów w okolicy — nieaktywny")
    poi_count = None




#__________________________________________________________________
#kodowanie inputu
#_________________________________________________________________

if predict_clicked:
    # Prepare input_data dict with your variables
    input_data = {
    "Miasto": city,
    "Powierzchnia": square_meters,
}

# Add only enabled and non-null values
if enable_rooms and rooms is not None:
    input_data["Liczba pokoi"] = rooms

if enable_floor_count and floor_count is not None:
    input_data["Liczba pięter w budynku"] = floor_count

if enable_floor and floor is not None:
    input_data["Piętro"] = floor

if enable_ownership and ownership is not None:
    input_data["Forma własności"] = ownership

if enable_has_parking and has_parking is not None:
    input_data["Miejsce parkingowe"] = boolean_map[has_parking]

if enable_has_balcony and has_balcony is not None:
    input_data["Balkon"] = boolean_map[has_balcony]

if enable_has_elevator and has_elevator is not None:
    input_data["Winda"] = boolean_map[has_elevator]

if enable_has_security and has_security is not None:
    input_data["Ochrona"] = boolean_map[has_security]

if enable_has_storage and has_storage is not None:
    input_data["Komórka lokatorska"] = boolean_map[has_storage]

if enable_build_year and build_year is not None:
    input_data["Rok budowy"] = build_year

if latitude is not None:
    input_data["Szerokość geograficzna"] = latitude

if longitude is not None:
    input_data["Długość geograficzna"] = longitude

if enable_poi_count and poi_count is not None:
    input_data["Liczba ważnych punktów w okolicy"] = poi_count





if predict_clicked:
    # Add enabled distance features only
    for label, value in distances.items():
        if value is not None:
            input_data[label] = value

    # Create DataFrame and apply one-hot encoding
    X = pd.DataFrame([input_data])
    st.write(X)
    X = pd.get_dummies(X)
    st.write(X)

    # Ensure all expected model features are present, fill missing using training defaults
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = fill_defaults.get(col, 0)  # fallback to 0 if not in defaults
    st.write("X po for")
    st.write(X)
    # Reorder columns to match model
    st.write("model.feature...:")
    st.write(model.feature_names_in_)

    X = X[model.feature_names_in_]

    st.write(X)


    # Predict and display
    prediction = model.predict(X)[0]
    price_placeholder.markdown(
        f"<div id='price-output'> Szacowana cena:<br><span>{prediction:,.0f} zł</span></div>",
        unsafe_allow_html=True
    )
else:
    price_placeholder.empty()


















#____________________________________________
#ch==sprawdzam input

if predict_clicked:
    # Add enabled distance features only
    for label, value in distances.items():
        if value is not None:
            input_data[label] = value

    # Create DataFrame and apply one-hot encoding
    X = pd.DataFrame([input_data])
    X = pd.get_dummies(X)

    st.subheader("X po get dummies")
    st.write(X)

    # Ensure all expected model features are present, fill missing using training defaults
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = fill_defaults.get(col, 0)

    # Reorder columns to match model
    X = X[model.feature_names_in_]

    #st.subheader("X po feature_names_in:")
    #st.write(X)

    # Predict and display
    prediction = model.predict(X)[0]
    price_placeholder.markdown(
        f"<div id='price-output'> Szacowana cena:<br><span>{prediction:,.0f} zł</span></div>",
        unsafe_allow_html=True
    )

    # 👇 Show debug info only after prediction
    st.markdown("---")
    st.subheader("📥 Dane wejściowe użyte do predykcji:")
    st.json(input_data)

    st.subheader("🧠 Dane wejściowe po kodowaniu (model-ready):")
    st.write(X)

else:
    price_placeholder.empty()











