import jax
import jax.numpy as jnp
from jax.scipy.special import expit

jax.config.update("jax_enable_x64", True)


def _predict(X, b):
    return expit(jnp.matmul(X, b))


def _firth_loglikelihood(X, y, b):
    pi = _predict(X, b)
    XW = _get_XW(X, pi)
    FIM = _fisher_info_mtx(XW)
    penalty = 0.5 * jnp.log(jnp.linalg.det(FIM))

    return -1 * (jnp.sum(y * jnp.log(pi) + (1 - y) * jnp.log(1 - pi)) + penalty)


def _fisher_info_mtx(XW):
    return jnp.matmul(XW.T, XW)


def _hat_diag(XW):
    Q = jnp.linalg.qr(XW, mode="reduced")[0]
    hat = jnp.einsum("ij,ij->i", Q, Q, precision="highest")
    return hat


def _get_XW(X, pi):
    rootW = jnp.sqrt(jnp.multiply(pi, 1 - pi))
    XW = rootW[:, jnp.newaxis] * X
    return XW


###################################################################################


@jax.jit
def _newton_iteration(X, y, b, step_limit):
    pi = _predict(X, b)
    XW = _get_XW(X, pi)
    FIM = _fisher_info_mtx(XW)
    hat = _hat_diag(XW)
    U = jnp.matmul(X.T, y - pi + jnp.multiply(hat, 0.5 - pi))
    b_new = b + jnp.linalg.lstsq(FIM, U)[0]
    # step halving
    ll = _firth_loglikelihood(X, y, b)
    new_ll = _firth_loglikelihood(X, y, b_new)
    carry = {
        "b_new": b_new,
        "b": b,
        "old_ll": ll,
        "new_ll": new_ll,
        "step": 0,
        "step_limit": step_limit,
    }
    res = jax.lax.while_loop(_step_condition, _step_halving, carry)
    return res


def _step_condition(carry):
    break_condition = carry["step"] == carry["step_limit"]
    return ~break_condition & (carry["new_ll"] > carry["old_ll"])


def _step_halving(carry):
    X, y = carry["X"], carry["y"]
    carry["b_new"] = carry["b"] + 0.5 * (carry["b_new"] - carry["b"])
    carry["new_ll"] = _firth_loglikelihood(X, y, carry["b_new"])
    carry["step"] += 1
    return carry


def _newton_cond(carry):
    return (
        jnp.linalg.norm(carry["b_new"] - carry["b"]) > carry["convergence_limit"]
    ) & (carry["step"] < carry["step_limit"])


def _newton_raphson(carry):
    X, y = carry["X"], carry["y"]
    carry["b"] = carry["b_new"]
    carry["b_new"] = _newton_iteration(X, y, carry["b_new"], carry["step_limit"])[
        "b_new"
    ]
    carry["step"] += 1
    return carry


@jax.jit
def fit_jax(
    X,
    y,
    step_limit=1000,
    convergence_limit=0.0001,
):
    device = "cpu"
    X = jax.device_put(X, jax.devices(device)[0])
    y = jax.device_put(y, jax.devices(device)[0])
    b = jnp.zeros(X.shape[1])
    b_new = _newton_iteration(X, y, b, step_limit)["b_new"]
    carry = {
        "b_new": b_new,
        "b": b,
        "old_ll": 0,
        "new_ll": 0,
        "step": 0,
        "step_limit": 1000,
        "convergence_limit": convergence_limit,
        "X": X,
        "y": y,
    }
    res = jax.lax.while_loop(_newton_cond, _newton_raphson, carry)

    return res["step"], res["b_new"], res["new_ll"]
