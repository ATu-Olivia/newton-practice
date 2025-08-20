# newton.py
from math import isfinite

def _grad(f, x, eps):
   
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)

def _hess(f, x, eps):
   
    return (f(x + eps) - 2.0 * f(x) + f(x - eps)) / (eps * eps)

def optimize(start, f, eps=1e-4, tol=1e-8, max_iter=100, verbose=False):
    
    x = float(start)
    last_x = x
    converged = False

    for it in range(1, max_iter + 1):
        g = _grad(f, x, eps)
        h = _hess(f, x, eps)

        # 防止除以非常小的二阶导（牛顿步会爆炸）
        if h == 0.0 or not isfinite(h):
            # 退火/降步：改用小步长的梯度下降作为保底
            step = 1e-2 * g
        else:
            step = g / h

        new_x = x - step

        if verbose:
            print(f"iter={it:3d}, x={x:.12g}, f(x)={f(x):.12g}, g={g:.4g}, h={h:.4g}, step={step:.4g}")

        # 收敛判据：位置变化很小
        if abs(new_x - x) < tol:
            x = new_x
            converged = True
            break

        last_x, x = x, new_x

    return {
        "x": x,
        "f": f(x),
        "n_iter": it,
        "converged": converged
    }
