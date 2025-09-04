import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DATA_PATH = r"C:\Users\ADMIN\Desktop\pima-indians-diabetes.csv"

if not (os.path.exists("scaler.pkl") and os.path.exists("diabetes_model.h5") and os.path.exists("history.pkl")):
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    model.save("diabetes_model.h5")
    pickle.dump((history.history, acc), open("history.pkl", "wb"))
else:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = keras.models.load_model("diabetes_model.h5")
    history, acc = pickle.load(open("history.pkl", "rb"))

root = tk.Tk()
root.title("🩺 Ứng dụng Chuẩn đoán Tiểu đường (Pima Indians)")
root.geometry("600x780")

labels = ["Số lần mang thai","Nồng độ Glucose","Huyết áp (mm Hg)","Độ dày da (mm)","Insulin","BMI","Chỉ số di truyền","Tuổi"]
entries = []
for text in labels:
    tk.Label(root, text=text, anchor="w", font=("Arial", 11)).pack(pady=2)
    entry = tk.Entry(root, font=("Arial", 11))
    entry.pack(pady=2, fill="x", padx=20)
    entries.append(entry)

tk.Label(root, text=f"🎯 Độ chính xác trên tập test: {acc:.2f}", font=("Arial", 12, "bold"), fg="blue").pack(pady=10)

def predict_diabetes():
    try:
        values = [float(e.get()) for e in entries]
        input_data = np.array([values])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0][0]
        if pred > 0.5:
            messagebox.showerror("Kết quả", f"⚠️ Nguy cơ TIỂU ĐƯỜNG cao (Xác suất {pred:.2f})")
        else:
            messagebox.showinfo("Kết quả", f"✅ Không có nguy cơ tiểu đường (Xác suất {pred:.2f})")
    except Exception as e:
        messagebox.showwarning("Lỗi", f"Vui lòng nhập số hợp lệ!\n\n{e}")

def reset_fields():
    for e in entries:
        e.delete(0, tk.END)

def show_big_plot():
    win = tk.Toplevel(root)
    win.title("📊 Biểu đồ Loss & Accuracy")
    fig = Figure(figsize=(6,4), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(history['loss'], label="Train Loss")
    ax1.plot(history['val_loss'], label="Val Loss")
    ax1.legend(); ax1.set_title("Loss")
    ax2.plot(history['accuracy'], label="Train Acc")
    ax2.plot(history['val_accuracy'], label="Val Acc")
    ax2.legend(); ax2.set_title("Accuracy")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

tk.Button(root, text="🔍 Dự đoán", command=predict_diabetes,
          font=("Arial", 12), bg="green", fg="white").pack(pady=10)
tk.Button(root, text="🧹 Reset", command=reset_fields,
          font=("Arial", 12), bg="gray", fg="white").pack(pady=5)
tk.Button(root, text="📊 Xem biểu đồ chi tiết", command=show_big_plot,
          font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

fig = Figure(figsize=(5,3), dpi=100)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(history['loss'], label="Train Loss")
ax1.plot(history['val_loss'], label="Val Loss")
ax1.legend(); ax1.set_title("Loss")
ax2.plot(history['accuracy'], label="Train Acc")
ax2.plot(history['val_accuracy'], label="Val Acc")
ax2.legend(); ax2.set_title("Accuracy")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(pady=20)

root.mainloop()
