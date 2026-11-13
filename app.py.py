# Simple Streamlit app for the Pet Adoption Speed project (ML Final Project).
# It has 2 pages:
#   1) Data Exploration  - look at the dataset and charts
#   2) Prediction        - enter a pet profile and get the adoption speed class
# Run it with:  streamlit run app.py
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 4)

# ---------------------------------------------------------------- settings
st.set_page_config(page_title="Pet Adoption Classifier", layout="centered")

speed_labels = {0: "Same day", 1: "First week", 2: "First month",
                3: "2nd-3rd month", 4: "Not adopted"}
type_map = {1: "Dog", 2: "Cat"}
gender_map = {1: "Male", 2: "Female", 3: "Mixed (group)"}
size_map = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra large"}
fur_map = {1: "Short", 2: "Medium", 3: "Long"}
yes_map = {1: "Yes", 2: "No", 3: "Not sure"}
health_map = {1: "Healthy", 2: "Minor injury", 3: "Serious injury"}
# reverse maps: name -> code
rev_type = {v: k for k, v in type_map.items()}
rev_gender = {v: k for k, v in gender_map.items()}
rev_size = {v: k for k, v in size_map.items()}
rev_fur = {v: k for k, v in fur_map.items()}
rev_yes = {v: k for k, v in yes_map.items()}
rev_health = {v: k for k, v in health_map.items()}

# ---------------------------------------------------------------- cached loading
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    breed = pd.read_csv("breed_labels.csv")
    color = pd.read_csv("color_labels.csv")
    state = pd.read_csv("state_labels.csv")
    return train, breed, color, state

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = json.load(open("feature_columns.json"))
    return model, scaler, feature_columns

train_df, breed_df, color_df, state_df = load_data()
model, scaler, feature_columns = load_model()

# ---------------------------------------------------------------- sidebar
page = st.sidebar.radio("Go to", ["Data Exploration", "Prediction"])
st.sidebar.caption("ML Final Project - Random Forest classifier")

# ================================================================ PAGE 1
if page == "Data Exploration":
    st.title("Data Exploration")

    # small summary numbers on top
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pets", len(train_df))
    c2.metric("Dogs", (train_df["Type"] == 1).sum())
    c3.metric("Cats", (train_df["Type"] == 2).sum())
    c4.metric("Columns", train_df.shape[1])

    st.subheader("First rows of the data")
    st.dataframe(train_df.head(10))

    st.subheader("Missing values")
    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        st.write("No missing values.")
    else:
        st.write(missing)

    # ---- chart 1: target distribution
    st.subheader("Adoption Speed distribution (the target)")
    counts = train_df["AdoptionSpeed"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar([speed_labels[i] for i in counts.index], counts.values, color="steelblue")
    ax.set_ylabel("Number of pets")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    # ---- chart 2: dogs vs cats + adoption speed by type
    st.subheader("Dogs vs Cats")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["Dog", "Cat"],
                [(train_df["Type"] == 1).sum(), (train_df["Type"] == 2).sum()],
                color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Number of pets")
    ct = pd.crosstab(train_df["Type"], train_df["AdoptionSpeed"], normalize="index") * 100
    ct.index = [type_map[i] for i in ct.index]
    bottom = pd.Series(0.0, index=ct.index)
    for col in ct.columns:
        axes[1].bar(ct.index, ct[col], bottom=bottom, label=str(col))
        bottom = bottom + ct[col]
    axes[1].set_title("Adoption speed % by type")
    axes[1].legend(title="Class", fontsize=7, ncol=5)
    st.pyplot(fig)

    # ---- chart 3: age distribution
    st.subheader("Age distribution (months)")
    fig, ax = plt.subplots()
    ax.hist(train_df["Age"].clip(upper=36), bins=36, color="teal")
    ax.set_xlabel("Age (months)")
    st.pyplot(fig)

    # ---- chart 4: correlation heatmap
    st.subheader("Correlation between numeric columns")
    num_cols = ["Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized",
                "Health", "Quantity", "Fee", "VideoAmt", "PhotoAmt", "AdoptionSpeed"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(train_df[num_cols].corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# ================================================================ PAGE 2
else:
    st.title("Prediction")
    st.write("Fill the pet profile below, then press **Predict** to see how fast "
             "this pet is expected to be adopted (class 0 = fastest ... class 4 = not adopted).")

    # options built from the label files
    dog_breeds = sorted(breed_df[breed_df["Type"] == 1]["BreedName"].unique())
    cat_breeds = sorted(breed_df[breed_df["Type"] == 2]["BreedName"].unique())
    color_names = sorted(color_df["ColorName"].unique())
    breed_to_id = dict(zip(breed_df["BreedName"], breed_df["BreedID"]))
    color_to_id = dict(zip(color_df["ColorName"], color_df["ColorID"]))
    state_names = sorted(state_df["StateName"].unique())
    state_to_id = dict(zip(state_df["StateName"], state_df["StateID"]))

    with st.form("pet_form"):
        col1, col2 = st.columns(2)

        ptype = col1.selectbox("Pet type", ["Dog", "Cat"])
        pname = col2.text_input("Pet name (leave empty if no name)")

        breeds = dog_breeds if ptype == "Dog" else cat_breeds
        pbreed1 = col1.selectbox("Main breed", breeds, index=breeds.index("Mixed Breed") if ptype == "Dog" else 0)
        pbreed2 = col2.selectbox("Second breed (optional)", ["None"] + breeds)

        page_num = col1.number_input("Age (months)", 0, 300, 3)
        pgender = col2.selectbox("Gender", list(gender_map.values()))

        pcolor1 = col1.selectbox("Color 1", color_names)
        pcolor2 = col2.selectbox("Color 2 (optional)", ["None"] + color_names)
        pcolor3 = col1.selectbox("Color 3 (optional)", ["None"] + color_names)

        psize = col2.selectbox("Maturity size", list(size_map.values()))
        pfur = col1.selectbox("Fur length", list(fur_map.values()))
        pvacc = col2.selectbox("Vaccinated", list(yes_map.values()))
        pdew = col1.selectbox("Dewormed", list(yes_map.values()))
        pster = col2.selectbox("Sterilized", list(yes_map.values()))
        phealth = col1.selectbox("Health", list(health_map.values()))

        pqty = col2.number_input("Quantity (pets in this listing)", 1, 20, 1)
        pfee = col1.number_input("Adoption fee (0 = free)", 0, 3000, 0)
        pstate = col2.selectbox("State", state_names)
        pphotos = col1.slider("Number of photos", 0, 30, 3)
        pvideos = col2.slider("Number of videos", 0, 8, 0)
        prescuer = col1.number_input("Rescuer posts (how many pets this rescuer listed)", 1, 500, 1)

        pdesc = st.text_area("Description text", "A lovely friendly pet looking for a home.")

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        # build one row with the SAME columns used in training
        row = {
            "Type": rev_type[ptype],
            "Age": page_num,
            "Breed1": breed_to_id[pbreed1],
            "Breed2": 0 if pbreed2 == "None" else breed_to_id[pbreed2],
            "Gender": rev_gender[pgender],
            "Color1": color_to_id[pcolor1],
            "Color2": 0 if pcolor2 == "None" else color_to_id[pcolor2],
            "Color3": 0 if pcolor3 == "None" else color_to_id[pcolor3],
            "MaturitySize": rev_size[psize],
            "FurLength": rev_fur[pfur],
            "Vaccinated": rev_yes[pvacc],
            "Dewormed": rev_yes[pdew],
            "Sterilized": rev_yes[pster],
            "Health": rev_health[phealth],
            "Quantity": pqty,
            "Fee": pfee,
            "State": state_to_id[pstate],
            "VideoAmt": pvideos,
            "PhotoAmt": pphotos,
            "HasName": 1 if pname.strip() else 0,
            "DescLength": len(pdesc),
            "RescuerPosts": prescuer,
        }
        X_row = pd.get_dummies(pd.DataFrame([row]),
                               columns=["Type", "Gender", "Color1", "Color2", "Color3",
                                        "MaturitySize", "FurLength", "Vaccinated",
                                        "Dewormed", "Sterilized", "Health", "State"])
        # align with the training columns (missing categories -> 0)
        X_row = X_row.reindex(columns=feature_columns, fill_value=0).astype(float)

        X_row_s = scaler.transform(X_row)
        pred = model.predict(X_row_s)[0]
        proba = model.predict_proba(X_row_s)[0]

        st.subheader("Result")
        st.success(f"Predicted adoption speed: class {pred} - {speed_labels[pred]}")

        st.subheader("Prediction probabilities")
        proba_series = pd.Series(proba, index=[f"{i} - {speed_labels[i]}" for i in range(5)])
        st.bar_chart(proba_series)

        with st.expander("What do the classes mean?"):
            st.write("| Class | Meaning |")
            st.write("|---|---|")
            st.write("| 0 | Adopted on the same day |")
            st.write("| 1 | Adopted in the first week |")
            st.write("| 2 | Adopted in the first month |")
            st.write("| 3 | Adopted in the 2nd-3rd month |")
            st.write("| 4 | Not adopted after 100 days |")
