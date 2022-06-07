import jax
import jax.numpy as jnp
from jax.scipy.special import expit

jax.config.update("jax_enable_x64", True)


@jax.jit
def fit_firth(
    X,
    y,
    max_iter=25,
    max_halfstep=1000,
    convergence_limit=0.0001,
):
    device = "cpu"
    mask = y.copy()
    X = jax.device_put(X, jax.devices(device)[0])
    y = jax.device_put(y, jax.devices(device)[0])
    b = jnp.zeros(X.shape[1])
    b_new = _newton_iteration(X, y, b, max_iter, mask)["b_new"]
    nr_carry = {
        "b_new": b_new,
        "b": b,
        "ll": 0,
        "iter": 1,
        "max_iter": max_iter,
        "max_halfstep": max_halfstep,
        "convergence_limit": convergence_limit,
        "X": X,
        "y": y,
        "mask": mask,
    }
    res = jax.lax.while_loop(_newton_cond, _newton_raphson, nr_carry)

    return res["iter"], res["b_new"], res["ll"]


def _newton_raphson(nr_carry):
    X, y = nr_carry["X"], nr_carry["y"]
    nr_carry["b"] = nr_carry["b_new"]
    iter = _newton_iteration(
        X, y, nr_carry["b_new"], nr_carry["max_halfstep"], nr_carry["mask"]
    )
    nr_carry["ll"] = iter["new_ll"]
    nr_carry["b_new"] = iter["b_new"]
    nr_carry["iter"] += 1
    return nr_carry


def _newton_cond(nr_carry):
    iter_limit = nr_carry["iter"] == nr_carry["max_iter"]
    converged = (
        jnp.linalg.norm(nr_carry["b_new"] - nr_carry["b"])
        < nr_carry["convergence_limit"]
    )
    return ~iter_limit & ~converged


@jax.jit
def _newton_iteration(X, y, b, max_halfstep, mask):
    pi = _predict(X, b, mask)
    XW = _get_XW(X, pi)
    FIM = jnp.matmul(XW.T, XW)
    hat = _hat_diag(XW)
    U_star = jnp.matmul(X.T, y - pi + jnp.multiply(hat, 0.5 - pi))
    b_new = b + jnp.linalg.lstsq(FIM, U_star)[0]
    pi_new = _predict(X, b_new, mask)
    ll = _firth_loglikelihood(X, y, pi, mask)
    new_ll = _firth_loglikelihood(X, y, pi_new, mask)
    # step halving
    step_carry = {
        "b_new": b_new,
        "b": b,
        "old_ll": ll,
        "new_ll": new_ll,
        "step": 0,
        "max_halfstep": max_halfstep,
        "X": X,
        "y": y,
        "mask": mask,
    }
    res = jax.lax.while_loop(_step_condition, _step_halving, step_carry)
    return res


def _step_condition(step_carry):
    step_limit = step_carry["step"] == step_carry["max_halfstep"]
    return ~step_limit & (step_carry["new_ll"] > step_carry["old_ll"])


def _step_halving(step_carry):
    X, y, mask = step_carry["X"], step_carry["y"], step_carry["mask"]
    step_carry["b_new"] = step_carry["b"] + 0.5 * (
        step_carry["b_new"] - step_carry["b"]
    )
    pi = _predict(X, step_carry["b_new"], mask)
    step_carry["new_ll"] = _firth_loglikelihood(X, y, pi, mask)
    step_carry["step"] += 1
    return step_carry


def _mask_array(array, mask, fill_val):
    return jnp.where(mask == -9, fill_val, array)


def _predict(X, b, mask):
    pi = expit(jnp.matmul(X, b))
    pi = _mask_array(pi, mask, 0)
    return pi


def _firth_loglikelihood(X, y, pi, mask):
    XW = _get_XW(X, pi)
    FIM = jnp.matmul(XW.T, XW)
    penalty = 0.5 * jnp.log(jnp.linalg.det(FIM))

    pi0 = _mask_array(pi, mask, 0)
    pi1 = _mask_array(pi, mask, 1)
    return -1 * (jnp.sum(y * jnp.log(pi1) + (1 - y) * jnp.log(1 - pi0)) + penalty)


def _hat_diag(XW):
    Q = jnp.linalg.qr(XW, mode="reduced")[0]
    hat = jnp.einsum("ij,ij->i", Q, Q, precision="highest")
    return hat


def _get_XW(X, pi):
    rootW = jnp.sqrt(jnp.multiply(pi, 1 - pi))
    XW = rootW[:, jnp.newaxis] * X
    return XW
