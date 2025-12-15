import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import mlflow
import mlflow.pytorch

# --- 1. Model Definition (RNN) ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)                 # (batch, seq, hidden)
        out = self.fc(out[:, -1, :])         # last timestep
        return out


# --- 2. Data Loading ---
def load_data(path):
    df = pd.read_csv(path)
    data = df.values.astype("float32")      

    train_input = torch.from_numpy(
        data[3:, :-1]
    ).unsqueeze(-1).float()                 

    train_target = torch.from_numpy(
        data[3:, -1]
    ).unsqueeze(-1).float()                 

    return train_input, train_target


# --- 3. Training Function ---
def train():
    DATA_PATH = "data/sine_wave.csv"
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 100

    mlflow.set_experiment("Basic_RNN_Project")

    with mlflow.start_run():
        # Log params
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("data_version", "v1")

        # Load data
        inputs, targets = load_data(DATA_PATH)

        # Model setup
        model = SimpleRNN(hidden_size=HIDDEN_SIZE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("Training started...")
        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
                mlflow.log_metric("loss", loss.item(), step=epoch)

        print("Training complete.")

        mlflow.pytorch.log_model(model, "rnn_model")
        print("Model saved to MLflow.")


if __name__ == "__main__":
    train()
