import numpy as np

model = np.load("anfis_model.npz")

def clean_input(moisture_deficit, atmospheric_temperature, atmospheric_humidity):
    """Validate and normalize the three model inputs."""
    inputs = np.asarray(
        [[moisture_deficit, atmospheric_temperature, atmospheric_humidity]],
        dtype=float,
    )
    if not np.isfinite(inputs).all():
        raise ValueError("All inputs must be finite numbers.")
    return (inputs - model["input_mean"]) / model["input_scale"]


def predict(inputs):
    """Return the predicted water delivery in L/m^2 for cleaned inputs."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.shape != (1, 3) or not np.isfinite(inputs).all():
        raise ValueError("inputs must be one cleaned row containing three finite values.")

    rules = model["rules"]
    log_weights = np.zeros((len(inputs), len(rules)))
    for feature in range(inputs.shape[1]):
        membership = rules[:, feature]
        difference = inputs[:, feature, None] - model["premise_centers"][feature, membership]
        log_weights -= 0.5 * (
            difference / model["premise_sigmas"][feature, membership]
        ) ** 2

    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    inputs_with_bias = np.column_stack((np.ones(len(inputs)), inputs))
    rule_outputs = inputs_with_bias @ model["consequents"].T
    return float(np.maximum((weights * rule_outputs).sum(axis=1), 0)[0])


if __name__ == "__main__":
    # Input order: moisture deficit (%), temperature (C), humidity (%).
    inputs = clean_input(10.0, 30.0, 70.0)
    prediction = predict(inputs)

    print(f"Predicted needed delivered: {prediction:.3f} L/m^2")
