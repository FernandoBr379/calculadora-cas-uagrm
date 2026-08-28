from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy as sp
import numpy as np  # pyrefly: ignore [missing-import]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalcRequest(BaseModel):
    expression: str
    variable: str = "x"
    operation: str  # 'simplify', 'limit', 'diff', 'integrate'
    point: str = "0"  # Punto para el límite x -> a

class PlotRequest(BaseModel):
    expression: str
    variable: str = "x"
    x_min: float = -10.0
    x_max: float = 10.0
    num_points: int = 300

@app.post("/api/eval")
def evaluate(data: CalcRequest):
    try:
        x = sp.Symbol(data.variable)
        expr = sp.sympify(data.expression)

        # 1. Simplificación algebraica exacta
        if data.operation == "simplify":
            res = sp.simplify(expr)

        # 2. Cálculo de límites (ordinarios y laterales)
        elif data.operation == "limit":
            pt = sp.sympify(data.point)
            res = sp.limit(expr, x, pt)

        # 3. Derivadas de funciones y de orden superior
        elif data.operation == "diff":
            res = sp.diff(expr, x)

        # 4. Integrales indefinidas / antiderivadas
        elif data.operation == "integrate":
            res = sp.integrate(expr, x)

        else:
            res = sp.simplify(expr)

        return {"status": "ok", "latex_result": sp.latex(res)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/plot")
def plot_function(data: PlotRequest):
    try:
        x = sp.Symbol(data.variable)
        expr = sp.sympify(data.expression)

        f_num = sp.lambdify(x, expr, modules=["numpy"])
        x_vals = np.linspace(data.x_min, data.x_max, data.num_points)
        y_vals = f_num(x_vals)

        y_list = [None if (np.isnan(v) or np.isinf(v)) else float(v) for v in y_vals]

        return {"status": "ok", "x": x_vals.tolist(), "y": y_list}

    except Exception as e:
        return {"status": "error", "message": str(e)}