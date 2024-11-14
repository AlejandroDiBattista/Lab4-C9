from fasthtml.common import *

app, rt = fast_app()

@dataclass
class Contacto:
    id:int
    nombre:str
    apellido:str
    edad:int

contactos = [
    Contacto(id=1,nombre='Juan', apellido='Perez', edad=30),
    Contacto(id=2,nombre='Maria', apellido='Lopez', edad=25),
    Contacto(id=3,nombre='Carlos', apellido='Gonzalez', edad=40),
]
 
def Formulario():
    return Form(
            Label('Nombre:',   Input(name='nombre', placeholder='Nombre')),
            Label('Apellido:', Input(name='apellido',placeholder='Apellido')),
            Label('Edad:',     Input(name='edad',placeholder='Edad')),
            Button('Agregar'),
            hx_post='/agregar',
            hx_target='#contactos',
            hx_swap='outerHTML'
        )

def MostrarContacto(contacto):
    return Li(
            H4(contacto.nombre, " ",
                Span(contacto.apellido, style='color:red;'),
            ),
            Div(f"Edad: {contacto.edad}"),
            A('Eliminar', 
              hx_delete=f'/borrar/{contacto.id}',
              hx_target='#contactos',
              hx_swap='outerHTML'),
            id=f"contacto-{contacto.id}"
    )

def ListaContactos():
    return Ul(
        *[MostrarContacto(contacto) for contacto in contactos],
        id='contactos'
    )

@rt('/')
def home():
    return Titled('Agregar Contacto',
        ListaContactos(),
        Formulario(),
    )

@rt('/borrar/{id}')
def delete(id:int):
    global contactos
    contactos = [c for c in contactos if c.id != id]
    return ListaContactos()

@rt('/agregar')
def post(contacto:Contacto):
    global contactos
    print('> Contacto', contacto)
    contacto.id = len(contactos)+1
    contactos.append(contacto)
    return ListaContactos()
serve()