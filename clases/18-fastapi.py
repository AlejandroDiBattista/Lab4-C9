from fastapi import FastAPI, HTTPException

from sqlmodel import SQLModel, Field, Session, create_engine, select

from typing import Optional, List

# Configuración de la base de datos
DATABASE_URL = "sqlite:///contacts.db"
engine = create_engine(DATABASE_URL, echo=True)

# Definición del modelo de contacto
class Contact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    nombre: str = Field(max_length=50)
    apellido: str
    telefono: str

    def nombre_completo(self):
        return f"{self.nombre} {self.apellido}"

    def guardar(self, engine):
        with Session(engine) as session:
            session.add(self)
            session.commit()
            session.refresh(self)
    
a = Contact(nombre="Juan", apellido="Pérez", telefono="123456789")
a.guardar(engine)

    
# Crear la tabla de contactos si no existe
SQLModel.metadata.create_all(engine)

# Agregar contactos de ejemplo si la tabla está vacía
with Session(engine) as session:
    if not session.exec(select(Contact)).all():
        contactos_ejemplo = [
            Contact(nombre="Juan", apellido="Pérez", telefono="123456789"),
            Contact(nombre="Ana", apellido="García", telefono="987654321"),
            Contact(nombre="Luis", apellido="Martínez", telefono="555555555")
        ]
        session.add_all(contactos_ejemplo)
        session.commit()

# Inicialización de la aplicación FastAPI
app = FastAPI()

# Endpoint para obtener todos los contactos
@app.get("/contacts/", response_model=List[Contact])
def get_contacts():
    with Session(engine) as session:
        contacts = session.exec(select(Contact)).all()
        contacts = session.exec(select(Contact).where(Contact.nombre == "Juan")).all()
    return contacts

# Endpoint para obtener un contacto por ID
@app.get("/contacts/{contact_id}", response_model=Contact)
def get_contact(contact_id: int):
    with Session(engine) as session:
        contact = session.get(Contact, contact_id)
        if not contact:
            raise HTTPException(status_code=404, detail="Contacto no encontrado")
    return contact

# Endpoint para crear un nuevo contacto
@app.post("/contacts/", response_model=Contact)
def create_contact(contact: Contact):
    with Session(engine) as session:
        session.add(contact)
        session.commit()
        session.refresh(contact)
    return contact

# Endpoint para actualizar un contacto
@app.put("/contacts/{contact_id}", response_model=Contact)
def update_contact(contact_id: int, contact_data: Contact):
    with Session(engine) as session:
        contact = session.get(Contact, contact_id)
        if not contact:
            raise HTTPException(status_code=404, detail="Contacto no encontrado")
        contact.nombre = contact_data.nombre
        contact.apellido = contact_data.apellido
        contact.telefono = contact_data.telefono
        session.commit()
        session.refresh(contact)
    return contact

# Endpoint para eliminar un contacto
@app.delete("/contacts/{contact_id}", response_model=Contact)
def delete_contact(contact_id: int):
    with Session(engine) as session:
        contact = session.get(Contact, contact_id)
        if not contact:
            raise HTTPException(status_code=404, detail="Contacto no encontrado")
        session.delete(contact)
        session.commit()
    return contact

nombre = "J';DELETE ALL FROM contact"
cmd = f"SELECT * FROM contact WHERE nombre = '{nombre}'"
print(cmd)