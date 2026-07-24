def discretize_state(state, max_queue_length=20, num_buckets=2):
    """
    Converts the true 16-value queue state into a bucketed version
    (e.g. "low"/"high" instead of an exact count). Keeps the Q-table
    a manageable size for tabular methods (Q-learning, SARSA).
    DQN does not use this -- it works on the full-precision state.
    """
    bucket_width = (max_queue_length + 1) / num_buckets
    bucketed = tuple(
        min(int(value // bucket_width), num_buckets - 1)
        for value in state
    )
    return bucketed
