import jax
import jax.numpy as jnp

# -------------------------------
# Scalar functions for each cost
# -------------------------------

def optimal_f_trade_0(p, pi, lambd=0.5, psi=0.3):
    return p / (2 * lambd) - pi

def optimal_f_trade_l2(p, pi, lambd=0.5, psi=0.3):
    return p / (2 * (lambd + psi)) + psi * pi / (lambd + psi) - pi

def optimal_f_trade_l1(p, pi, lambd=0.5, psi=0.3):
    lower = -psi + 2 * lambd * pi
    upper =  psi + 2 * lambd * pi

    val1 = (p + psi) / (2 * lambd) - pi
    val2 = 0.0
    val3 = (p - psi) / (2 * lambd) - pi

    return jnp.where(
        p <= lower,
        val1,
        jnp.where(
            p < upper,
            val2,
            val3
        )
    )

# -------------------------------
# JITed scalar versions
# -------------------------------

optimal_f_trade_0_jit  = jax.jit(optimal_f_trade_0)
optimal_f_trade_l2_jit  = jax.jit(optimal_f_trade_l2)
optimal_f_trade_l1_jit  = jax.jit(optimal_f_trade_l1)

# -------------------------------
# Vectorized versions (batched)
# -------------------------------

optimal_f_trade_0_batch = jax.jit(jax.vmap(optimal_f_trade_0))
optimal_f_trade_l2_batch = jax.jit(jax.vmap(optimal_f_trade_l2))
optimal_f_trade_l1_batch = jax.jit(jax.vmap(optimal_f_trade_l1))

# -------------------------------
# Test section
# -------------------------------

if __name__ == "__main__":
    # Example scalar inputs
    p_scalar = 0.1
    pi_scalar = 0.0

    print("=== Scalar evaluations ===")
    print("Trade 0 :", optimal_f_trade_0_jit(p_scalar, pi_scalar))
    print("Trade L2:", optimal_f_trade_l2_jit(p_scalar, pi_scalar))
    print("Trade L1:", optimal_f_trade_l1_jit(p_scalar, pi_scalar))

    # Example batched inputs
    p_batch = jnp.array([0.1, 0.2, -0.3, 0.5])
    pi_batch = jnp.array([0.0, 0.05, -0.1, 0.2])

    print("\n=== Batched evaluations ===")
    print("Trade 0 batch :", optimal_f_trade_0_batch(p_batch, pi_batch))
    print("Trade L2 batch:", optimal_f_trade_l2_batch(p_batch, pi_batch))
    print("Trade L1 batch:", optimal_f_trade_l1_batch(p_batch, pi_batch))

    print("\n=== JIT + VMAP warmup ===")
    for _ in range(3):
        _ = optimal_f_trade_l2_batch(p_batch, pi_batch)
    print("JIT compiled and ready.")
