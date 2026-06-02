"""Pure-numpy synthetic sequence generators for each task.

Every generator fn has signature:
    gen_episode(rng, T, action_dim, obs_dim, **kwargs) -> dict

Returns dict with keys:
  - state: (T, obs_dim) float64
  - action: (T, action_dim) float64
  - info: dict with task-specific metadata
"""

import numpy as np


def linear_ar(rng, T, action_dim=4, obs_dim=16, spectral_radius=0.95,
              action_coupling=0.3, noise=0.01, **kw):
    """x_{t+1} = A x_t + B a_t + ε

    A is random with tunable spectral radius. B maps actions to state changes.
    """
    A = rng.randn(obs_dim, obs_dim) * (spectral_radius / np.sqrt(obs_dim))
    B = rng.randn(obs_dim, action_dim) * action_coupling
    x = rng.randn(obs_dim) * 0.1
    states = np.empty((T, obs_dim), dtype=np.float64)
    actions = rng.randn(T, action_dim).astype(np.float64)
    for t in range(T):
        states[t] = x
        x = A @ x + B @ actions[t] + rng.randn(obs_dim) * noise
    return dict(state=states, action=actions,
                info=dict(spectral_radius=spectral_radius, noise=noise))


def nback(rng, T, action_dim=4, obs_dim=16, n_back=3, vocab_size=8, **kw):
    """Discrete token at step t must be remembered at step t+n_back.

    The action at step t encodes a token (discrete 0..vocab_size-1).
    The state at step t+n_back encodes whether it matches the token.
    """
    tokens = rng.randint(0, vocab_size, size=T)
    one_hot = np.eye(vocab_size, dtype=np.float64)
    states = np.zeros((T, obs_dim), dtype=np.float64)
    actions = np.zeros((T, action_dim), dtype=np.float64)
    for t in range(T):
        actions[t, 0] = tokens[t]
        actions[t, 1] = tokens[t] / vocab_size
        states[t, :vocab_size] = one_hot[tokens[t]]
        if t >= n_back:
            states[t, vocab_size] = float(tokens[t] == tokens[t - n_back])
    return dict(state=states, action=actions,
                info=dict(n_back=n_back, vocab_size=vocab_size))


def delayed_copy(rng, T, action_dim=4, obs_dim=16, delay=10,
                 pattern_len=4, vocab_size=6, **kw):
    """Show a pattern at step 0, then at step delay prompt the model to recall.

    The pattern is a sequence of pattern_len discrete tokens shown at t=0.
    The prompt is a special marker at t=delay.
    The target is the pattern repeated.
    """
    pattern = rng.randint(0, vocab_size, size=pattern_len)
    one_hot = np.eye(vocab_size, dtype=np.float64)
    states = np.zeros((T, obs_dim), dtype=np.float64)
    actions = np.zeros((T, action_dim), dtype=np.float64)

    # Show pattern at beginning
    for i in range(min(pattern_len, T)):
        if i < vocab_size:
            states[i, :vocab_size] = one_hot[pattern[i]]
        actions[i, 0] = -1.0  # show marker

    # Blank period
    prompt_step = min(delay, T - pattern_len - 1)
    if prompt_step > pattern_len:
        actions[prompt_step, 0] = 1.0  # prompt marker

    # Recall period
    recall_start = prompt_step + 1
    for i in range(min(pattern_len, T - recall_start)):
        if i < vocab_size:
            states[recall_start + i, :vocab_size] = one_hot[pattern[i]]
        actions[recall_start + i, 0] = -2.0  # recall marker

    return dict(state=states, action=actions,
                info=dict(delay=delay, pattern_len=pattern_len, vocab_size=vocab_size))


def slowfast(rng, T, action_dim=4, obs_dim=16, slow_tau=50, fast_tau=3,
             slow_amplitude=0.8, fast_amplitude=0.2, noise=0.01, **kw):
    """x_t = slow_t + fast_t, two timescales.

    Slow component: AR(1) with correlation time slow_tau.
    Fast component: AR(1) with correlation time fast_tau.
    Action modulates the fast component.
    """
    def ar1_process(length, tau):
        """Generate AR(1) process with given correlation time."""
        phi = np.exp(-1.0 / tau)
        proc = np.empty((length, obs_dim // 2))
        z = rng.randn(obs_dim // 2)
        for t in range(length):
            z = phi * z + np.sqrt(1 - phi ** 2) * rng.randn(obs_dim // 2)
            proc[t] = z
        return proc

    slow = ar1_process(T, slow_tau) * slow_amplitude
    fast = ar1_process(T, fast_tau) * fast_amplitude
    actions = rng.randn(T, action_dim).astype(np.float64) * 0.5

    # Action modulates fast component
    fast += actions[:, :1] * fast_amplitude * 0.3

    states = np.concatenate([slow + fast], axis=1)
    return dict(state=states, action=actions,
                info=dict(slow_tau=slow_tau, fast_tau=fast_tau, noise=noise))


def chaotic(rng, T, action_dim=4, obs_dim=16, lorenz_rho=28.0,
            lorenz_sigma=10.0, lorenz_beta=8.0 / 3.0, dt=0.02,
            action_coupling=1.0, noise=0.0, **kw):
    """Lorenz attractor driven by actions.

    The full Lorenz state is 3D, but we embed it into obs_dim using a random
    projection. Actions perturb the system.
    """
    # Lorenz equations
    def lorenz_step(x, y, z, ax, ay, az):
        dx = lorenz_sigma * (y - x) + ax * action_coupling
        dy = x * (lorenz_rho - z) - y + ay * action_coupling
        dz = x * y - lorenz_beta * z + az * action_coupling
        return x + dx * dt, y + dy * dt, z + dz * dt

    # Random embedding from 3D to obs_dim
    P = rng.randn(obs_dim, 3).astype(np.float64)
    P /= np.linalg.norm(P, axis=1, keepdims=True)

    states = np.empty((T, obs_dim), dtype=np.float64)
    actions = rng.randn(T, action_dim).astype(np.float64)
    x, y, z = 1.0, 1.0, 1.0

    for t in range(T):
        embedded = (P @ np.array([x, y, z])) + rng.randn(obs_dim) * noise
        states[t] = embedded
        x, y, z = lorenz_step(x, y, z, actions[t, 0], actions[t, 1], actions[t, 2])
        if noise:
            x += rng.randn() * noise
            y += rng.randn() * noise
            z += rng.randn() * noise

    return dict(state=states, action=actions,
                info=dict(lorenz_rho=lorenz_rho, noise=noise))


TASKS = {
    "linear_ar": (linear_ar, dict(obs_dim=16, action_dim=4)),
    "nback": (nback, dict(obs_dim=16, action_dim=4, n_back=3, vocab_size=8)),
    "delayed_copy": (delayed_copy, dict(obs_dim=16, action_dim=4, delay=10, pattern_len=4, vocab_size=6)),
    "slowfast": (slowfast, dict(obs_dim=16, action_dim=4, slow_tau=50, fast_tau=3)),
    "chaotic": (chaotic, dict(obs_dim=16, action_dim=4)),
}
