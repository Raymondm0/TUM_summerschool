import pandas as pd
import numpy as np
from numpy.typing import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def rising_edge_idx(data:ArrayLike, interval:int) -> int:
    rising_edge = 4
    for i in range(len(data) - 4):
        start = data[i]
        end = data[i + 4]
        if end - start >= interval:
            rising_edge += 1
            break
        else:
            rising_edge += 1
    return rising_edge

def falling_edge_idx(data:ArrayLike, interval:int) -> int:
    falling_edge = 0
    for i in range(len(data) - 4):
        start = data[i]
        end = data[i + 4]
        if start - end >= interval:
            break
        else:
            falling_edge += 1
    return falling_edge

def get_product_data() -> pd.DataFrame:
    """
    Returns:
        raw labeled dataset in dataframe.
        "dataset": qualified: 1; unqualified: 0
        "audio": audio noise: 1; explicit: 0
        "error_type": no error: 0; unqualified for oil: 1; for offset: 2; for tilt: 3
    """
    labels_path = "C:\\Users\\14631\\Desktop\\TUMint Summer School\\PQAM\\02_Data\\02_Data\\0001_Database.xlsx"
    df = pd.read_excel(labels_path, skiprows=10)
    dataset = df["Dataset"]
    audio = df["Audio"]
    error_type = df["Forced Error Type"]
    labeled_dataset = pd.DataFrame({
        "dataset": dataset,
        "audio": audio,
        "error_type": error_type
    })

    product_dict = {"iO": 1, "niO": 0}
    audio_dict = {"Yes": 1, "No": 0}
    error_dict = {np.NAN: 0, "oil": 1, "offset": 2, "Tilt": 3}
    for i in range(len(labeled_dataset)):
        labeled_dataset.loc[i, "dataset"] = product_dict[labeled_dataset["dataset"][i]]
        labeled_dataset.loc[i, "audio"] = audio_dict[labeled_dataset["audio"][i]]
        labeled_dataset.loc[i, "error_type"] = error_dict[labeled_dataset["error_type"][i]]

    return labeled_dataset

def get_raw_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Power [W]"] = np.array(df["Current [A]"] * df["Voltage [V]"])
    df = df.drop(["Wire [m/min]"])

    return df

def get_features() -> pd.DataFrame:
    """
    Returns:
        features derived from Current-time, Voltage-time, Power-time graphs relatively
    """
    electric_path = "C:\\Users\\14631\\Desktop\\TUMint Summer School\\PQAM\\02_Data\\02_Data\\02_Weldqas"
    file_list = [f"{i}" for i in range(169, 429)]
    features = []

    for i in range(len(file_list)):
        path = electric_path+"\\"+file_list[i]+".csv"
        df = get_raw_data(path)
        t = np.array(df["Time [s]"])
        I = np.array(df["Current [A]"])
        U = np.array(df["Voltage [V]"])
        P = np.array(df["Power [W]"])

        df_cut = df.iloc[::20]
        I_w = np.array(df_cut["Current [A]"]).reshape(-1,1)
        I_rise, I_fall = rising_edge_idx(I_w, 220) * 20, falling_edge_idx(I_w, 220) * 20

        U_w = np.array(df_cut["Voltage [V]"]).reshape(-1,1)
        U_rise, U_fall = rising_edge_idx(U_w, 8) * 20, falling_edge_idx(U_w, 15) * 20

        I_welding, U_welding, P_welding = np.array(I[I_rise:I_fall+1]).reshape(-1, 1), np.array(U[U_rise:U_fall+1]).reshape(-1, 1), np.array(P[I_rise:U_fall]+1).reshape(-1, 1)

        I_var = np.var(I_welding, axis=0).item()
        I_mean = np.mean(I_welding, axis=0).item()
        U_var = np.var(U_welding, axis=0).item()
        U_mean = np.mean(U_welding, axis=0).item()
        rise_diff = I_rise - U_rise # phase differences of current and voltage
        P_max = P_welding[np.argmax(P_welding)].item()
        P_mean = np.mean(P_welding, axis=0).item()
        P_var = np.var(I_welding, axis=0).item()
        # max_arc_temperature = P_welding[np.argmax(P_welding)].item() # max power
        # arc_stability = np.var(U_welding, axis=0).item() # variance of voltage
        weld_duration = t[U_fall] - t[I_rise]
        heating_slope = float((P_max - U_welding[0])/(t[np.argmax(P_welding) + I_rise] - t[I_rise]))
        features.append([I_var, U_var, P_max, rise_diff, weld_duration, heating_slope])

    features = np.array(features).T
    features = pd.DataFrame({
        "I_var":features[0],
        "U_var":features[1],
        "P_max":features[2],
        "rise_diff":features[3],
        "weld_duration":features[4],
        "heating_slope":features[5]
    })
    return features

def show_derived_data(set_num=1):
    path = f"C:\\Users\\14631\\Desktop\\TUMint Summer School\\PQAM\\02_Data\\02_Data\\02_Weldqas\\{set_num+168}.csv"
    df = pd.read_csv(path)
    df["Power [W]"] = np.array(df["Current [A]"] * df["Voltage [V]"])
    df = df.drop(["Wire [m/min]"])

    t = np.array(df["Time [s]"])
    I = np.array(df["Current [A]"])
    U = np.array(df["Voltage [V]"])
    P = np.array(df["Power [W]"])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), )
    fig.suptitle(f"dataset {set_num+168}")

    ax[0].plot(t, I)
    ax[0].set_xlabel('t/s')
    ax[0].set_ylabel('I/A')
    ax[0].set_title('I-t')

    ax[1].plot(t, U)
    ax[1].set_xlabel('t/s')
    ax[1].set_ylabel('U/V')
    ax[1].set_title('U-t')

    ax[2].plot(t, P)
    ax[2].set_xlabel('t/s')
    ax[2].set_ylabel('P/W')
    ax[2].set_title('P-t')

    df_critical = df.iloc[::20]
    I_c = np.array(df_critical["Current [A]"]).reshape(-1, 1)
    I_rise, I_fall = rising_edge_idx(I_c, 220) * 20, falling_edge_idx(I_c, 220) * 20
    ax[0].plot(t[I_rise:I_fall + 1], I[I_rise:I_fall + 1], color="orange", label="I working_zone")
    ax[0].legend()

    U_c = np.array(df_critical["Voltage [V]"]).reshape(-1, 1)
    U_rise, U_fall = rising_edge_idx(U_c, 8) * 20, falling_edge_idx(U_c, 15) * 20
    ax[1].plot(t[U_rise:U_fall + 1], U[U_rise:U_fall + 1], color="orange", label="U working_zone")
    ax[1].legend()

    ax[2].plot(t[I_rise:U_fall + 1], P[I_rise:U_fall + 1], color="orange", label="P working_zone")
    ax[2].legend()

    plt.show()

def split_data(x_data:ArrayLike, y_data:ArrayLike):
    X0, X1, X2, X3 = [], [], [], []
    Y0, Y1, Y2, Y3 = [], [], [], []
    for i in range(len(x_data)):
        if y_data[i] == 0:
            X0.append(x_data[i])
            Y0.append(y_data[i])
        elif y_data[i] == 1:
            X1.append(x_data[i])
            Y1.append(y_data[i])
        elif y_data[i] == 2:
            X2.append(x_data[i])
            Y2.append(y_data[i])
        else:
            X3.append(x_data[i])
            Y3.append(y_data[i])
    X0_train, X0_test, X1_train, X1_test, Y0_train, Y0_test, Y1_train, Y1_test = train_test_split(
        X0, X1, Y0, Y1, train_size=0.8, shuffle=True
    )
    X2_train, X2_test, X3_train, X3_test, Y2_train, Y2_test, Y3_train, Y3_test = train_test_split(
        X2, X3, Y2, Y3, train_size=0.8, shuffle=True
    )
    X_train = np.concatenate((X0_train, X1_train, X2_train, X3_train), axis=0)
    X_test = np.concatenate((X0_test, X1_test, X2_test, X3_test), axis=0)
    Y_train = np.concatenate((Y0_train, Y1_train, Y2_train, Y3_train), axis=0)
    Y_test = np.concatenate((Y0_test, Y1_test, Y2_test, Y3_test), axis=0)

    return X_train, X_test, Y_train, Y_test