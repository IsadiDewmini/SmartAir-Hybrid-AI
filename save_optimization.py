
import matplotlib.pyplot as plt

confidence = [0.05, 0.10, 0.12, 0.14,
              0.16, 0.18, 0.20, 0.25]

mae = [7.08, 0.75, 0.75, 1.58,
       1.75, 2.50, 3.25, 4.25]

plt.figure(figsize=(9, 5))

plt.plot(confidence, mae, marker="o")

plt.xlabel("Confidence Threshold")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("SmartAir YOLO Confidence Optimization")

plt.grid(True)

plt.savefig("smartair_confidence_optimization.png",
            dpi=300)

plt.show()

print("Optimization graph saved!")
