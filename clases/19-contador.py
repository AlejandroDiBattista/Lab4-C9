from fasthtml.common import *

app, rt = fast_app()

contador = 10 

def Inc(cantidad):
    return P(
        A(f'+ {cantidad}', 
          hx_put=f'/incrementar/{cantidad}',
          hx_target='#contador'
        )
    ),

def Contador():
    color = 'red' if contador < 0 else 'green'
    return H4(f"Contador: {contador}", 
              id='contador',
              style=f"color:{color};"
              )

@rt('/')
def get():
    return Titled('Mi contador',
        Contador(),
        Inc(1),Inc(3), 
        Inc(-1),Inc(-3), 
        
    )

@rt('/incrementar/{cantidad}')
def put(cantidad:int):
    global contador
    contador += cantidad
    return Contador()

serve()