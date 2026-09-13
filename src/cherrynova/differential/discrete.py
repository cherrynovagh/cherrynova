def diff(t, x):
    # Check that both input arrays have the same length
    if len(t) != len(x):
        raise ValueError("Input arrays t and x must have the same length")

    # stores the discrete derivative values
    v = []

    # Loop through the data starting from the second point
    for k in range(1, len(t)):
        # Compute the discrete derivative using the formula:
        # v(t) = (x(t_k) - x(t_k-1)) / (t_k - t_k-1)
        derivative = (x[k] - x[k - 1]) / (t[k] - t[k - 1])
        v.append(derivative)

    return v
